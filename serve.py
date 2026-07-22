#!/usr/bin/env python3
"""Local dev server for the fbx-memetics viewer + subject-activation backend.

Serves viewer/ statically (replaces `python3 -m http.server`) and exposes:
  GET  /api/registry            → active/potential subject registry
  POST /api/activate {subject}  → start activation (runs the per-subject pipeline)
                                   returns {job}
  GET  /api/job/<job>           → {status: running|done|error, log, subject}

Activation runs in a background thread (narrative generation is an LLM call, ~30-60s).
Run: python3 serve.py   →  http://localhost:8811/showcase.html
"""
import glob
import hashlib
import hmac
import io
import json
import os
import re
import secrets as pysecrets
import threading
import time
import traceback
import uuid
import zipfile

from flask import Flask, request, jsonify, redirect, send_from_directory, send_file

import pipeline
import instance_config

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.join(HERE, "viewer")

# load .env at import so the gate + API key work under gunicorn too
for _line in (open(os.path.join(HERE, ".env")) if os.path.exists(os.path.join(HERE, ".env")) else []):
    if "=" in _line and not _line.strip().startswith("#"):
        _k, _v = _line.strip().split("=", 1)
        os.environ.setdefault(_k, _v)

app = Flask(__name__, static_folder=None)

# ---------------- password gate ----------------
# With STORYDECK_PASSWORD set (deployments), everything except the landing
# page and the unlock endpoint requires a signed auth cookie. No corpus data,
# viewer page, or API response is reachable without it. The token is derived
# from a per-boot secret, so a restart invalidates all sessions.
AUTH_COOKIE = "sd_auth"
_boot_secret = pysecrets.token_hex(32)
PUBLIC_PATHS = {"/", "/api/unlock", "/favicon.ico", "/robots.txt"}


def _auth_token():
    return hmac.new(_boot_secret.encode(), b"storydeck-unlocked", hashlib.sha256).hexdigest()


def _authed():
    return hmac.compare_digest(request.cookies.get(AUTH_COOKIE, ""), _auth_token())


@app.before_request
def _gate():
    if not os.environ.get("STORYDECK_PASSWORD"):
        return None                     # no password configured (local dev): open
    if request.path in PUBLIC_PATHS or _authed():
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "locked"}), 401
    return redirect("/")


@app.post("/api/unlock")
def unlock():
    pw = os.environ.get("STORYDECK_PASSWORD") or ""
    got = ((request.get_json(silent=True) or {}).get("password") or "")
    time.sleep(0.6)                     # brute-force damper
    if pw and hmac.compare_digest(got.encode(), pw.encode()):
        resp = jsonify({"ok": True})
        resp.set_cookie(AUTH_COOKIE, _auth_token(), httponly=True, samesite="Lax",
                        secure=request.headers.get("X-Forwarded-Proto") == "https",
                        max_age=7 * 86400)
        return resp
    return jsonify({"error": "wrong password"}), 403


@app.get("/robots.txt")
def robots():
    return "User-agent: *\nDisallow: /\n", 200, {"Content-Type": "text/plain"}
jobs = {}          # job_id -> {status, log, subject}
job_lock = threading.Lock()
activate_lock = threading.Lock()   # serialize activations (single API budget)


# ---------------- query: chat over the whole mined corpus ----------------
_qctx = {"text": None, "legend": None, "parts": None}
_qctx_lock = threading.Lock()

HIST_INTRO = instance_config.fill_program((
    "You are the Storydeck oracle for the {PROGRAM}, "
    "situated IN TIME: today is {cutoff}. Below is the complete mined "
    "corpus of every meeting UP THROUGH TODAY — each meeting's summary and grounded "
    "(subject, relation, object) triples, in chronological order, each block headed "
    "with a tag like [#42] and each of its triples numbered like (3). Nothing after "
    "{cutoff} exists yet: do not reference, assume, or predict later events as fact. "
    "You may extrapolate about the future, clearly framed as expectation from "
    "today's vantage.\n"
    "GROUNDING RULES — the PAST is grounded, the FUTURE is yours to extrapolate:\n"
    "1. PAST AND PRESENT (through {cutoff}): every factual assertion must be backed "
    "by specific mined triples, cited inline right after the claim as [#42.3] "
    "(meeting 42, triple 3), or several as [#42.3][#42.7][#51.0]. Bare [#42] only "
    "for meeting-level claims grounded in the summary. If no triple or summary "
    "supports a past-tense claim, DO NOT assert it — say the data doesn't show it. "
    "Never invent names, dates, decisions, or causality about what has happened.\n"
    "2. THE FUTURE (after {cutoff}): questions about what WILL happen are welcome "
    "and you are expected to extrapolate — this is not a grounding violation. Mark "
    "predictions as predictions ('my expectation', 'the trajectory suggests'), and "
    "anchor each one in cited current evidence — the [#i.j] facts whose trajectory "
    "you are extrapolating. Never present a prediction as an accomplished fact.\n"
    "3. Keep the two registers visibly distinct: cited fact vs. framed expectation.\n"
    "Be concrete: say who said/did what and when. Cite liberally. Transcripts are "
    "machine-transcribed, so proper nouns are sometimes garbled.\n\n"))


def query_context():
    """The full cohort context (chronological summaries + bare triples + strands),
    built once and cached in memory. ~710k tokens; served to Fable with a
    cache_control block so repeat questions hit the prompt cache."""
    with _qctx_lock:
        if _qctx["text"] is None:
            import gen_fabric
            strands = []
            for f in sorted(glob.glob(os.path.join(HERE, "data", "facts", "STORY_*__isaacson.md"))):
                key = os.path.basename(f)[len("STORY_"):-len("__isaacson.md")]
                raw = re.sub(r"^<!--SRC .*?-->\n", "", open(f).read(), flags=re.S)
                m = re.search(r"^STRAND:\s*(.+)$", raw, flags=re.M)
                if m:
                    strands.append(f"- {key}: {m.group(1).strip()}")
            corpus, legend = gen_fabric.lean_corpus(numbered=True)
            _qctx["legend"] = legend
            # per-meeting split for historic cutoffs (compare mode)
            parts = re.split(r"\n(?=### \[#)", corpus)
            _qctx["parts"] = [
                (m.group(1) if (m := re.match(r"### \[#\d+\] \[(\d{4}-\d{2}-\d{2})", p)) else "", p)
                for p in parts]
            _qctx["text"] = instance_config.fill_program(
                "You are the Storydeck oracle for the {PROGRAM} "
                "({SPAN_FULL}). Below is the complete mined corpus: every "
                "meeting's summary and grounded (subject, relation, object) triples, in "
                "chronological order, followed by one-line STRAND arc compressions per subject. "
                "Each meeting block is headed with a tag like [#42] and each of its triples is "
                "numbered like (3).\n"
                "GROUNDING RULES — you must stay inside this data:\n"
                "1. Every FACTUAL assertion must be backed by specific mined triples: cite them "
                "inline right after the claim as [#42.3] (meeting 42, triple 3), or several as "
                "[#42.3][#42.7][#51.0]. Before asserting, find the triple; the citation is the "
                "evidence, not decoration.\n"
                "2. Use a bare meeting tag [#42] only for meeting-level claims (that a meeting "
                "happened, its overall topic) grounded in the summary rather than one triple.\n"
                "3. If no triple or summary supports a claim, DO NOT assert it — say the data "
                "doesn't show it. Never invent names, dates, decisions, or causality.\n"
                "4. When you synthesize across meetings, label it explicitly and keep it "
                "clearly separate from cited fact. Questions about the future beyond this "
                "corpus welcome extrapolation — mark predictions as predictions, anchored "
                "in the cited facts whose trajectory you are extending.\n"
                "Be concrete: say who said/did what and when. Cite liberally — every "
                "substantive sentence should carry at least one tag. Transcripts are machine-"
                "transcribed, so proper nouns are sometimes garbled.\n\n"
                ) + (corpus
                + "\n\n## STRANDS (per-subject arc compressions)\n" + "\n".join(strands))
        return _qctx["text"]


def historic_context(cutoff):
    """Corpus truncated at cutoff (inclusive) with an in-time intro; no strands
    (subject arcs span the whole program and would leak the future)."""
    query_context()
    sel = [p for d, p in _qctx["parts"] if not d or d <= cutoff]
    return HIST_INTRO.format(cutoff=cutoff) + "\n".join(sel)


@app.get("/api/query/legend")
def query_legend():
    query_context()
    return jsonify(_qctx["legend"])


@app.post("/api/query")
def query():
    """Streaming chat: prepend the whole cohort context (prompt-cached) and run
    the conversation on Fable."""
    from flask import Response, stream_with_context
    body = request.get_json(force=True)
    messages = body.get("messages") or []
    if not messages:
        return jsonify({"error": "no messages"}), 400
    cutoff = (body.get("cutoff") or "").strip()
    if cutoff and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff):
        return jsonify({"error": "bad cutoff"}), 400
    ctx = historic_context(cutoff) if cutoff else query_context()

    def gen():
        try:
            import anthropic
            for line in open(os.path.join(HERE, ".env")):
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k, v)
            client = anthropic.Anthropic(max_retries=2, timeout=900)
            # Opus 4.8 primary: Fable intermittently returned EMPTY text bodies on
            # this ~710k prompt (thinking-only); Opus is reliable here and half the
            # price. Retry-on-empty kept as a safety net.
            final = None
            for attempt in range(3):
                emitted = 0
                with client.messages.stream(
                    model="claude-opus-4-8", max_tokens=8000,
                    system=[{"type": "text", "text": ctx,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=messages,
                ) as st:
                    for txt in st.text_stream:
                        emitted += len(txt)
                        yield txt
                    final = st.get_final_message()
                if emitted > 0:
                    break
                print(f"[query] empty Fable body (attempt {attempt + 1}), retrying…", flush=True)
            u = final.usage
            cr = getattr(u, "cache_read_input_tokens", 0) or 0
            cw = getattr(u, "cache_creation_input_tokens", 0) or 0
            cost = (u.input_tokens * 5 + cw * 6.25 + cr * 0.5 + u.output_tokens * 25) / 1e6
            if emitted == 0:
                yield "(the model returned an empty response three times — try re-asking)"
            yield f"\n\x1e{{\"in\": {u.input_tokens}, \"cache_read\": {cr}, \"cache_write\": {cw}, \"out\": {u.output_tokens}, \"cost\": {cost:.2f}}}"
        except Exception as e:
            yield f"\n\x1e{{\"error\": \"{str(e)[:200]}\"}}"

    return Response(stream_with_context(gen()), mimetype="text/plain")


JUDGE_PROMPT = (
    "You are the Storydeck judge. The same question was asked to two oracles over the "
    "same meeting corpus: HISTORIC knows only meetings through {cutoff}; FULL knows the "
    "whole program.\n\nQUESTION:\n{question}\n\nHISTORIC ANSWER (knowledge ends {cutoff}):\n"
    "{hist}\n\nFULL-CORPUS ANSWER:\n{full}\n\n"
    "Synthesize briefly:\n"
    "**Overlap** — what both answers agree on.\n"
    "**Divergence** — where they part: what the historic vantage missed, got wrong, or "
    "left open that the full corpus resolves.\n"
    "**What changed** — one or two lines on what the divergence reveals about what "
    "actually happened after {cutoff}.\n"
    "Be concrete and under ~180 words. Preserve any [#i] / [#i.j] citation tags from "
    "the answers you reference.")


@app.post("/api/query/judge")
def query_judge():
    """Streaming synthesis of a compare-mode pair: overlap vs divergence between
    the historic-cutoff answer and the full-corpus answer. No corpus context —
    just the two answers, so it's fast and cheap."""
    from flask import Response, stream_with_context
    body = request.get_json(force=True)
    qn = (body.get("question") or "").strip()
    hist = (body.get("historic") or "").strip()
    full = (body.get("full") or "").strip()
    cutoff = (body.get("cutoff") or "?").strip()
    if not (qn and hist and full):
        return jsonify({"error": "need question, historic, full"}), 400
    prompt = JUDGE_PROMPT.format(cutoff=cutoff, question=qn, hist=hist, full=full)

    def gen():
        try:
            import anthropic
            for line in open(os.path.join(HERE, ".env")):
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k, v)
            client = anthropic.Anthropic(max_retries=2, timeout=300)
            with client.messages.stream(
                model="claude-opus-4-8", max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            ) as st:
                for txt in st.text_stream:
                    yield txt
                final = st.get_final_message()
            u = final.usage
            cost = (u.input_tokens * 5 + u.output_tokens * 25) / 1e6
            yield f"\n\x1e{{\"in\": {u.input_tokens}, \"cache_read\": 0, \"cache_write\": 0, \"out\": {u.output_tokens}, \"cost\": {cost:.2f}}}"
        except Exception as e:
            yield f"\n\x1e{{\"error\": \"{str(e)[:200]}\"}}"

    return Response(stream_with_context(gen()), mimetype="text/plain")


@app.get("/api/stories.zip")
def stories_zip():
    """Zip every subject's Isaacson story as clean prose (SRC map + [[i]] citation
    tags stripped) for download."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(glob.glob(os.path.join(HERE, "data", "facts", "STORY_*__isaacson.md"))):
            key = os.path.basename(f)[len("STORY_"):-len("__isaacson.md")]
            raw = open(f).read()
            raw = re.sub(r"^<!--SRC .*?-->\n", "", raw, flags=re.S)   # drop machine legend
            raw = re.sub(r"\s*\[\[[\d,\s]+\]\]", "", raw)             # drop citation tags
            z.writestr(f"{key}.md", raw.strip() + "\n")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="fbx-stories.zip")


@app.get("/api/registry")
def registry():
    p = os.path.join(VIEWER, "data", "registry.json")
    return (open(p).read(), 200, {"Content-Type": "application/json"}) if os.path.exists(p) \
        else jsonify({"subjects": {}, "n_active": 0, "n_potential": 0})


def _run_activation(job_id, query, novel):
    def log(m):
        with job_lock:
            jobs[job_id]["log"].append(m)
    try:
        with activate_lock:
            if novel:
                # resurface an arbitrary typed term from the corpus (grep → grounded
                # nodes → narrative); falls back to the extracted subject if it is one
                log(f"searching the corpus for '{query}'…")
                cid = pipeline.activate_novel(query)
            else:
                log(f"resolving '{query}'…")
                cid, how = pipeline.resolve_subject(query)
                if not cid:
                    raise RuntimeError(how)
                log(f"→ {cid} ({how}); generating story…")
                pipeline.activate(cid)
            with job_lock:
                jobs[job_id]["subject"] = cid
            log("done")
        with job_lock:
            jobs[job_id]["status"] = "done"
    except Exception as e:
        traceback.print_exc()
        with job_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["log"].append(f"error: {e}")


@app.post("/api/activate")
def activate():
    body = request.json or {}
    query = (body.get("subject") or "").strip()
    if not query:
        return jsonify({"error": "no subject"}), 400
    job_id = uuid.uuid4().hex[:12]
    with job_lock:
        jobs[job_id] = {"status": "running", "log": [], "subject": None}
    threading.Thread(target=_run_activation, args=(job_id, query, bool(body.get("novel"))),
                     daemon=True).start()
    return jsonify({"job": job_id})


def _run_relationship(job_id, a, b, force):
    def log(m):
        with job_lock:
            jobs[job_id]["log"].append(m)
    try:
        with activate_lock:
            log(f"inferring how {a} and {b} relate…")
            key = pipeline.compute_relationship(a, b, force=force)
            with job_lock:
                jobs[job_id]["key"] = key
            log("done")
        with job_lock:
            jobs[job_id]["status"] = "done"
    except Exception as e:
        traceback.print_exc()
        with job_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["log"].append(f"error: {e}")


@app.post("/api/relationship")
def relationship():
    body = request.json or {}
    a, b = (body.get("a") or "").strip(), (body.get("b") or "").strip()
    if not a or not b:
        return jsonify({"error": "need two subjects a and b"}), 400
    job_id = uuid.uuid4().hex[:12]
    with job_lock:
        jobs[job_id] = {"status": "running", "log": [], "key": None}
    threading.Thread(target=_run_relationship, args=(job_id, a, b, bool(body.get("force"))),
                     daemon=True).start()
    return jsonify({"job": job_id})


@app.post("/api/deactivate")
def deactivate():
    query = ((request.json or {}).get("subject") or "").strip()
    if not query:
        return jsonify({"error": "no subject"}), 400
    try:
        with activate_lock:
            cid = pipeline.deactivate(query)
        return jsonify({"ok": True, "subject": cid})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.get("/api/job/<job>")
def job_status(job):
    with job_lock:
        j = jobs.get(job)
    return jsonify(j or {"status": "unknown"})


@app.get("/")
def root():
    """Locked deployments land on the public landing page; unlocked sessions
    (and passwordless local dev) go straight to the experience."""
    page = "showcase.html"
    if os.environ.get("STORYDECK_PASSWORD") and not _authed():
        page = "landing.html"
    resp = send_from_directory(VIEWER, page)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/<path:path>")
def static_files(path):
    resp = send_from_directory(VIEWER, path)
    if path.endswith(".html"):
        # viewer pages change often mid-session; a stale cached page silently
        # breaks new citation formats / API contracts
        resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    print("fbx-memetics dev server → http://localhost:8811/showcase.html")
    app.run(host="127.0.0.1", port=8811, debug=False, threaded=True)
