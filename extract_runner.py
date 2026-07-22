#!/usr/bin/env python3
"""Full-corpus subject + beat extraction via the Anthropic API.

For each indexed transcript (excluding derived per-team splits):
  stage 1: subjects  -> data/subjects/<id>.json   (v1 prompt, API port)
  stage 2: beats     -> data/beats/<id>.json      (v2 prompt, consumes stage-1 output)

- model: claude-opus-4-8, adaptive thinking, streaming
- resumable: existing output files are skipped (the 5 pilot transcripts keep
  their subagent-produced outputs)
- usage ledger: data/usage_log.csv (per call: tokens + $) for reimbursement
- concurrency: THREADS workers; SDK retries 429/5xx, plus loop-level backoff

Usage: python3 extract_runner.py [--dry-run] [--limit N]
"""
import argparse
import csv
import json
import os
import instance_config as _ic
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
VAULT = os.environ.get("FBX_VAULT", os.path.join(HERE, "data", "vault"))
COHORT = os.path.join(VAULT, "cohort")
MODEL = "claude-opus-4-8"
IN_PRICE, OUT_PRICE = 5.00 / 1e6, 25.00 / 1e6  # $/token
THREADS = 6
MAX_TOKENS = 64000

# --- shared program context (prepended to every call) ---------------------

def build_context():
    """Program context for the extraction prompts — the prose comes from
    instance.json (program.context_note); a team_profiles.csv is appended
    when the instance provides one."""
    ctx = "## Program context (for your reference)\n\n" + _ic.program()["context_note"] + "\n"
    tp = os.path.join(COHORT, "team_profiles.csv")
    if os.path.exists(tp):
        ctx += "\nTeam roster (team_profiles.csv):\n" + open(tp).read() + "\n"
    return ctx

CONTEXT = build_context()

SUBJECTS_PROMPT = _ic.fill_program("""You are performing subject extraction on a meeting transcript from the {PROGRAM} program ({SEASON}), for a quantitative-memetics study of how ideas propagate through the program's language over time.

{context}
## Transcript metadata
{meta}

## What to extract

**Subjects**: units of meaning-making that the conversation is *about*. Types:

- `team` — cohort teams / startups (see roster above)
- `product` — named products, tools, apps, repos (theirs or external)
- `technology` — technical systems/primitives (e.g. TEE, dstack, MEV, MPC, attestation, LLM routing)
- `concept` — ideas, frames, theories, mechanisms (e.g. "coordination OS", "agent economy", "retroactive attribution", "local-first privacy")
- `meme` — coined terms, in-jokes, recurring phrases with internal currency (e.g. a program in-joke, a recurring ritual name)
- `person` — a person *as a subject of discussion* (NOT merely as a speaker — speaker names are noisy metadata; only include a person if what they are/do/think is itself discussed)
- `org` — external organizations (companies, labs, protocols, ...)
- `event` — named events (demo day, hackathons, salons as referenced events)

Prioritize **specific, recurring, program-internal** units over generic topics. "agent memory" as a design problem they name and argue about = subject. "talking about lunch" = not a subject. Skip pure logistics (scheduling, mic checks).

## What to record per subject

- `id`: kebab-case canonical slug (stable across transcripts — use the most standard name)
- `display`: human-readable canonical name
- `type`: one of the types above
- `aliases`: surface forms seen in THIS transcript (for later frequency counting), including abbreviations and misheard/mis-transcribed variants
- `definition`: if the transcript contains a **definitional / meaning-making move** for this subject (someone explains what it IS, coins it, reframes it), record `{{"line": N, "quote": "..."}}` of the best one. Omit if none.
- `mentions`: every line where the subject is **substantively engaged**. For each: `{{"line": N, "quote": "...", "kind": "..."}}` where `quote` is an EXACT substring copied verbatim from that line (10-200 chars, the most meaning-bearing span) and `kind` is one of `definition` | `discussion` | `passing`. Do not pad with `passing` mentions of every token occurrence.

## Transcript (line numbers are canonical — cite them exactly)

{transcript}

## Output

Respond with ONLY a JSON object (no code fences, no commentary):
{{"transcript": "{tid}", "extractor_version": "v1-api", "subjects": [...], "notes": "1-3 sentences on anything structurally odd"}}

Quality bar: quotes MUST be verbatim substrings of the cited line (machine-verified; mismatches dropped). Expect roughly 10-40 subjects for a typical 1-2h meeting; err toward precision.""")

def _fill_tracked(t):
    tp = _ic.program().get("tracked_person", {"label": "?", "note": ""})
    return t.replace("{TRACKED_NOTE}", tp.get("note", "")).replace("{TRACKED}", tp.get("label", "?"))


BEATS_PROMPT = _fill_tracked(_ic.fill_program("""You are segmenting a meeting transcript from the {PROGRAM} ({SEASON}) into a hierarchical tree of **narrative beats**, for a quantitative-memetics study. A "beat" is a contiguous stretch of conversation with one narrative identity — e.g. "AV setup", "Smithers weekly update", "Tina monologue on fundraising strategy", "dstack live demo", "Q&A: attestation".

{context}
## Transcript metadata
{meta}

## Already-extracted subjects for this transcript (reuse these id slugs when tagging beats)
{subjects}

## Beat tree rules

- Top-level beats **tile the whole file**: first starts at line 1, last ends at line {nlines}, contiguous, non-overlapping, in order.
- A beat may have `children` that tile ITS range the same way. Max depth 3.
- Leaf beats should be coherent dive-in units — typically 5-80 lines. No hundreds of tiny leaves; no 400-line undifferentiated blobs when structure is visible.
- `kind`: one of `presentation` | `demo` | `qa` | `discussion` | `monologue` | `interview` | `planning` | `logistics` | `other`. Use `logistics` for AV setup, scheduling, mic checks.

## {TRACKED}

{TRACKED_NOTE}
- Where speaker labels name them, identification is trivial.
- Where labels are generic (`Speaker N`) or absent, infer from context (coordinator-register monologues, being addressed by name, program-strategy framing). Be honest about uncertainty.
- Per beat set `tina`: "dominant" | "present" | "absent" | "unknown" (the field name is fixed; it means the tracked person). Also set top-level `tina_evidence`: 1-2 sentences on HOW the tracked person was identified in this file, or note they don't appear.

## Transcript (line numbers are canonical)

{transcript}

## Output

Respond with ONLY a JSON object (no code fences, no commentary):
{{"transcript": "{tid}", "version": "v2-beats-api", "tina_evidence": "...", "beats": [{{"label": "...", "kind": "...", "start": 1, "end": N, "summary": "...", "subjects": ["slug"], "speakers": ["..."], "tina": "...", "children": [...]}}], "new_subjects": [{{"id": "...", "display": "...", "type": "..."}}]}}

Labels are for a human browsing a tree — specific and short. Tiling is machine-verified."""))

# --- infra -----------------------------------------------------------------

log_lock = threading.Lock()
print_lock = threading.Lock()
USAGE_CSV = os.path.join(HERE, "data", "usage_log.csv")


def log_usage(tid, stage, usage):
    cost = usage.input_tokens * IN_PRICE + usage.output_tokens * OUT_PRICE
    with log_lock:
        new = not os.path.exists(USAGE_CSV)
        with open(USAGE_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "transcript", "stage", "model", "input_tokens",
                            "cache_read_tokens", "output_tokens", "cost_usd"])
            w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), tid, stage, MODEL,
                        usage.input_tokens, usage.cache_read_input_tokens or 0,
                        usage.output_tokens, f"{cost:.4f}"])
    return cost


def say(msg):
    with print_lock:
        print(msg, flush=True)


def numbered(lines):
    return "\n".join(f"{i+1}\t{l}" for i, l in enumerate(lines))


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def call(client, tid, stage, prompt):
    import anthropic
    for attempt in range(5):
        try:
            with client.messages.stream(
                model=MODEL, max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                msg = stream.get_final_message()
            cost = log_usage(tid, stage, msg.usage)
            if msg.stop_reason == "max_tokens":
                say(f"!! {tid} {stage}: truncated at max_tokens")
            text = next((b.text for b in msg.content if b.type == "text"), "")
            return parse_json(text), cost
        except anthropic.RateLimitError:
            wait = min(60, 5 * 2 ** attempt)
            say(f".. {tid} {stage}: 429, backing off {wait}s")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(10 * (attempt + 1))
            else:
                raise
        except json.JSONDecodeError as e:
            say(f"!! {tid} {stage}: bad JSON ({e}), retrying")
            if attempt >= 1:
                raise
    raise RuntimeError(f"{tid} {stage}: exhausted retries")


def meta_block(r):
    keys = ["title", "date", "meeting_type", "team", "teams_covered", "audience",
            "participants", "recording_app", "speaker_labels", "notes"]
    return "\n".join(f"- {k}: {r[k]}" for k in keys if r.get(k, "").strip())


def process(client, r, dry):
    tid = r["id"]
    spath = os.path.join(HERE, "data", "subjects", f"{tid}.json")
    bpath = os.path.join(HERE, "data", "beats", f"{tid}.json")
    if os.path.exists(spath) and os.path.exists(bpath):
        return 0.0
    with open(os.path.join(VAULT, "transcripts", r["path"]), encoding="utf-8", errors="replace") as f:
        raw = f.read()
    lines = raw.splitlines()
    if len(raw.split()) < 15:               # empty/stub transcript (some are one long line)
        return 0.0
    tnum, meta = numbered(lines), meta_block(r)
    total = 0.0
    if dry:
        say(f"DRY {tid}: {len(lines)} lines, ~{len(tnum)//4} transcript tokens")
        return 0.0
    if not os.path.exists(spath):
        subjects, cost = call(client, tid, "subjects", SUBJECTS_PROMPT.format(
            context=CONTEXT, meta=meta, transcript=tnum, tid=tid))
        total += cost
        with open(spath, "w") as f:
            json.dump(subjects, f)
        say(f"ok {tid} subjects: {len(subjects.get('subjects', []))} (${cost:.2f})")
    with open(spath) as f:
        subjects = json.load(f)
    slim = [{"id": s["id"], "display": s["display"], "type": s["type"]}
            for s in subjects.get("subjects", [])]
    if not os.path.exists(bpath):
        beats, cost = call(client, tid, "beats", BEATS_PROMPT.format(
            context=CONTEXT, meta=meta, subjects=json.dumps(slim), transcript=tnum,
            tid=tid, nlines=len(lines)))
        total += cost
        with open(bpath, "w") as f:
            json.dump(beats, f)
        say(f"ok {tid} beats: {len(beats.get('beats', []))} top-level (${cost:.2f})")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # load .env
    with open(os.path.join(HERE, ".env")) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)

    with open(os.path.join(VAULT, "index.csv")) as f:
        rows = [r for r in csv.DictReader(f) if not r["derived_from"].strip()]
    rows.sort(key=lambda r: r["date"])
    if args.limit:
        rows = [r for r in rows
                if not (os.path.exists(os.path.join(HERE, "data", "subjects", f'{r["id"]}.json'))
                        and os.path.exists(os.path.join(HERE, "data", "beats", f'{r["id"]}.json')))][:args.limit]
    say(f"work list: {len(rows)} transcripts, {THREADS} workers, model {MODEL}")

    client = None
    if not args.dry_run:
        import anthropic
        client = anthropic.Anthropic(max_retries=3, timeout=600)   # abort+retry hung requests

    done = fail = 0
    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = {ex.submit(process, client, r, args.dry_run): r["id"] for r in rows}
        for fut in __import__("concurrent.futures", fromlist=["as_completed"]).as_completed(futures):
            tid = futures[fut]
            try:
                total_cost += fut.result()
                done += 1
            except Exception as e:
                fail += 1
                say(f"FAIL {tid}: {type(e).__name__}: {e}")
    say(f"DONE: {done} ok, {fail} failed, total ${total_cost:.2f}")


if __name__ == "__main__":
    main()
