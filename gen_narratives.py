#!/usr/bin/env python3
"""Generate per-subject narratives for the showcase's 3-level interaction model.

For every canonical subject (>=2 transcripts): one Opus 4.8 call over a compact
digest of its hit beats (labels + beat summaries + sample mention quotes,
chronological) producing:
  - arc:    the long-arc story of the idea across the program (rhizome level)
  - per_tx: {transcript_id: 1-2 sentence note on what THIS conversation
             revealed/added about the subject, relative to the arc} (tree level)

Outputs data/narratives/<subject>.json (resumable) then compile with
--compile into viewer/data/narratives.json.
Usage: python3 gen_narratives.py [--limit N] [--compile]
"""
import argparse
import csv
import glob
import json
import os
import re
import threading
import time
import instance_config as _ic
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-opus-4-8"
IN_PRICE, OUT_PRICE = 5.00 / 1e6, 25.00 / 1e6
THREADS = 6
OUTDIR = os.path.join(HERE, "data", "narratives")

log_lock = threading.Lock()
print_lock = threading.Lock()


def say(m):
    with print_lock:
        print(m, flush=True)


def log_usage(subject, usage):
    cost = usage.input_tokens * IN_PRICE + usage.output_tokens * OUT_PRICE
    with log_lock:
        with open(os.path.join(HERE, "data", "usage_log.csv"), "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), subject, "narrative",
                                    MODEL, usage.input_tokens, usage.cache_read_input_tokens or 0,
                                    usage.output_tokens, f"{cost:.4f}"])
    return cost


def build_digests():
    """subject -> list of per-transcript {tx,date,title,mtype,beats,quotes}, built
    from the showcase hits so PERSON subjects include beats they SPOKE in (voice),
    not only beats where they're discussed. Capped for prompt size."""
    import person_voice
    CANON = json.load(open(os.path.join(HERE, "data", "canonical_vocab.json")))["canon"]
    show = json.load(open(os.path.join(HERE, "viewer", "data", "showcase.json")))
    TX = show["transcripts"]
    SUB = show["subjects"]

    tx_cache = {}
    def tx_doc(ti):
        if ti not in tx_cache:
            tx_cache[ti] = json.load(open(os.path.join(HERE, "viewer", "data", f"{TX[ti]['id']}.json")))
        return tx_cache[ti]

    def beat_at(ti, path):
        node = TX[ti]["beats"][path[0]]
        for idx in path[1:]:
            node = node["c"][idx]
        return node

    digests = {}
    for cid, meta in SUB.items():
        voice = set(meta.get("voice", []))
        disp = meta["display"]
        # discussed-mention quotes per line, from the raw extraction
        by_tx = {}
        for h in meta["hits"]:
            by_tx.setdefault(h[0], []).append(h)
        for ti, hs in sorted(by_tx.items(), key=lambda kv: TX[kv[0]]["date"]):
            d = tx_doc(ti)
            lines = d["lines"]; speakers = d.get("speakers") or []
            matched = person_voice.resolve_speakers(disp, {x for x in speakers if x}) if meta["type"] == "person" else set()
            # canonical mention lines in this tx
            ment_lines = []
            for sub in d["subjects"]:
                if CANON.get(sub["id"], sub["id"]) == cid:
                    ment_lines += [(m["line"], m["quote"]) for m in sub["mentions"]]
            beats = []
            for h in hs:
                key = "/".join(map(str, h))
                b = beat_at(ti, h[1:])
                is_voice = key in voice
                q = ""
                if is_voice and matched:
                    spoken = [lines[i] for i in range(b["s"] - 1, min(b["e"], len(lines)))
                              if i < len(speakers) and speakers[i] in matched and len(lines[i].split()) > 4]
                    if spoken:
                        q = max(spoken, key=len)[:180]
                else:
                    inb = [qt for ln, qt in ment_lines if b["s"] <= ln <= b["e"]]
                    if inb:
                        q = inb[0][:160]
                beats.append({"label": b["l"], "summary": (b.get("sm") or "")[:180],
                              "voice": is_voice, "w": b.get("w", 0), "quote": q})
            # cap per transcript: prefer higher-word / quoted beats
            beats.sort(key=lambda x: (bool(x["quote"]), x["w"]), reverse=True)
            beats = beats[:4]
            digests.setdefault(cid, []).append({
                "tx": TX[ti]["id"], "date": TX[ti]["date"], "title": TX[ti]["title"][:80],
                "mtype": TX[ti]["mtype"], "beats": beats})
    return digests, SUB


PROMPT = _ic.fill_program("""You are writing narrative syntheses for a quantitative-memetics study of the {PROGRAM} ({SPAN_ACTIVE}). Below is the complete chronological digest of every conversation in which the subject **{display}** ({stype}) was substantively discussed: per conversation, the narrative beats it appeared in (labels + beat summaries) and sample verbatim quotes (ASR — proper nouns often garbled). Beats marked **[SPOKE]** are ones the subject (a person) actually spoke in — the «quote» is their own words; unmarked beats are where the subject is discussed by others. For a person, their story is mostly what they SAID and did across the program, not just when others mentioned them.

{story_so_far}## Digest
{digest}

## Task
Write, as JSON (no fences, no commentary):
{{
 "arc_beats": [
   {{"text": "One sentence (or two short ones) of the long-arc story.", "tx": ["<transcript id>", ...]}},
   ...
 ],
 "per_tx": {{
   "<transcript id>": "1-2 sentences: what THIS conversation revealed, added, or changed about the subject — positioned relative to the arc (e.g. first appearance, pivot, consolidation, echo). Cover the pivotal conversations — always every id referenced in arc_beats, plus other notable moments — but you need not write one for every routine appearance; aim for up to ~50 of the most significant. Spread them across the whole timeline, including the most recent conversations.",
   ...
 }}
}}

If a "STORY SO FAR" section is present, treat it as your previous synthesis: preserve its framing and continuity, and EXTEND it to incorporate the (possibly newly added) conversations rather than contradicting it — refine wording only where the new evidence warrants. `arc_beats` rules: 6-10 beats that, read in order, tell the long-arc story of the subject across the program — where it came from, how it evolved, turning points, where it ended up. Concrete, naming conversations/moments; written for a reader browsing a visualization. Each beat's `tx` lists the transcript ids (from the digest, verbatim) that ground THAT beat — the conversations a reader should be shown when they highlight it. Use [] only for pure connective tissue.

CRITICAL — TEMPORAL COVERAGE: the digest is in chronological order and may span many months. Your arc MUST reach the subject's MOST RECENT conversations — do not stop partway and treat the latest weeks as a tail. The final 2-3 beats must ground in the latest portion of the digest (roughly its last third by date). Spread grounding so every active stretch of the timeline — beginning, middle, AND end — is reachable through some beat's `tx`. A story that stops a month before the digest ends is wrong.""")


def gen_one(client, cid, entries, meta):
    import anthropic
    lines = []
    for e in entries:
        parts = []
        for b in e["beats"]:
            tag = "[SPOKE] " if b.get("voice") else ""
            seg = tag + b["label"] + (f" — {b['summary']}" if b["summary"] else "")
            if b.get("quote"):
                seg += f'  «{b["quote"]}»'
            parts.append(seg)
        lines.append(f"[{e['date']}] {e['title']} ({e['mtype']}) :: id={e['tx']}\n  " + "\n  ".join(parts))
    digest = "\n".join(lines)
    m = meta.get(cid, {})
    prior = os.path.join(OUTDIR, f"{cid}.json")
    story_so_far = ""
    if os.path.exists(prior):
        try:
            pd = json.load(open(prior))
            arc = pd.get("arc") or " ".join(b.get("text", "") for b in pd.get("arc_beats", []))
            if arc.strip():
                story_so_far = f"## STORY SO FAR (your previous synthesis — extend, don't contradict)\n{arc}\n\n"
        except Exception:
            pass
    prompt = PROMPT.format(display=m.get("display", cid), stype=m.get("type", "concept"),
                           digest=digest, story_so_far=story_so_far)
    for attempt in range(4):
        try:
            with client.messages.stream(model=MODEL, max_tokens=48000,
                                        thinking={"type": "disabled"},
                                        messages=[{"role": "user", "content": prompt}]) as st:
                msg = st.get_final_message()
            cost = log_usage(cid, msg.usage)
            text = next((b.text for b in msg.content if b.type == "text"), "")
            if not text:
                say(f".. {cid}: empty text, stop={msg.stop_reason}, out={msg.usage.output_tokens}")
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            data = json.loads(text[text.find("{"):text.rfind("}") + 1])
            assert "arc_beats" in data and "per_tx" in data
            # safety net: models tend to end the arc at the narrative climax (Demo Day)
            # and leave the latest conversations ungrounded. Force the final beat to
            # ground the ~10 most-recent cards so the arc always reaches the present
            # (see the recurring "arc isn't using the later cards" bug).
            edates = {e["tx"]: e["date"] for e in entries}
            if edates and data["arc_beats"]:
                order = sorted(edates, key=lambda t: edates[t])
                grounded = {t for b in data["arc_beats"] for t in b.get("tx", [])}
                recent = [t for t in reversed(order) if t not in grounded][:10]
                lb = data["arc_beats"][-1]
                lb.setdefault("tx", [])
                for t in sorted(recent, key=lambda t: edates[t]):
                    if t not in lb["tx"]:
                        lb["tx"].append(t)
            data["arc"] = " ".join(b["text"] for b in data["arc_beats"])
            with open(os.path.join(OUTDIR, f"{cid}.json"), "w") as f:
                json.dump(data, f)
            return cost
        except anthropic.RateLimitError:
            time.sleep(min(60, 5 * 2 ** attempt))
        except (json.JSONDecodeError, AssertionError) as e:
            if attempt >= 2:
                raise
            say(f".. {cid}: bad JSON, retry")
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(10)
            else:
                raise
    raise RuntimeError(f"{cid}: exhausted retries")


def compile_out():
    out = {}
    for p in glob.glob(os.path.join(OUTDIR, "*.json")):
        out[os.path.basename(p)[:-5]] = json.load(open(p))
    dst = os.path.join(HERE, "viewer", "data", "narratives.json")
    with open(dst, "w") as f:
        json.dump(out, f)
    print(f"compiled {len(out)} narratives -> {dst} ({os.path.getsize(dst)//1024}KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    if args.compile:
        compile_out()
        return
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(HERE, ".env")) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)
    import anthropic
    client = anthropic.Anthropic(max_retries=3)
    digests, meta = build_digests()
    if args.only:
        work = [(args.only, digests[args.only])]
    else:
        work = [(cid, e) for cid, e in digests.items()
                if not os.path.exists(os.path.join(OUTDIR, f"{cid}.json"))]
    work.sort(key=lambda x: -len(x[1]))
    if args.limit:
        work = work[:args.limit]
    say(f"work: {len(work)} subjects, {THREADS} workers")
    done = fail = 0
    total = 0.0
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = {ex.submit(gen_one, client, cid, e, meta): cid for cid, e in work}
        for fu in as_completed(futs):
            cid = futs[fu]
            try:
                total += fu.result()
                done += 1
                if done % 25 == 0:
                    say(f"... {done}/{len(work)} (${total:.2f})")
            except Exception as e:
                fail += 1
                say(f"FAIL {cid}: {type(e).__name__}: {e}")
    say(f"DONE: {done} ok, {fail} failed, ${total:.2f}")
    compile_out()


if __name__ == "__main__":
    main()
