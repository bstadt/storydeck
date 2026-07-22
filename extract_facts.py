#!/usr/bin/env python3
"""Mine a single transcript into a summary + (subject, relation, object) triples.

Pilot for the fact/relationship extraction layer (the "shifting ontology"): each
transcript -> a compressed structured representation (summary + typed triples)
that can be fed en masse as context for narrative generation, routed into the
entanglement graph (relational triples = edges), and handed to a verifier agent
(claim triples = things to check against sources).

Usage:
  python3 extract_facts.py <transcript.md> [--model claude-opus-4-8]
Writes data/facts/<transcript_id>.json and prints a human-readable view.
"""
import argparse
import json
import os
import re
import sys
import instance_config as _ic

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "data", "facts")


def load_env():
    with open(os.path.join(HERE, ".env")) as f:
        for line in f:
            s = line.strip()
            if "=" in s and not s.startswith("#"):
                k, v = s.split("=", 1)
                os.environ.setdefault(k, v)


# a "Speaker Name  HH:MM" (or HH:MM:SS) line — Otter docx-export turn header
_TS_HEADER = re.compile(r"^.{0,60}?\s\d{1,2}:\d{2}(?::\d{2})?\s*$")


def number_turns(text):
    """Split the transcript into speaker turns and prefix each with [T<n>] so
    triples can cite exactly where in the original transcript they are grounded.
    Handles two vault formats: (a) markdown `**Speaker**` turns separated by
    blank lines, and (b) raw Otter exports where each turn starts with a
    `Speaker Name  HH:MM` header and there are no blank-line separators."""
    lines = text.split("\n")
    hdrs = [i for i, l in enumerate(lines) if l.strip() and _TS_HEADER.match(l.strip())]
    chunks = []
    if len(hdrs) >= 5:  # timestamped Otter format: split at each speaker header
        for j, start in enumerate(hdrs):
            end = hdrs[j + 1] if j + 1 < len(hdrs) else len(lines)
            block = "\n".join(lines[start:end]).strip()
            if block:
                chunks.append(block)
    if len(chunks) < 3:  # fall back to blank-line split (markdown format)
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    turns = {i: c for i, c in enumerate(chunks, 1)}
    numbered = "\n\n".join(f"[T{i}] {c}" for i, c in turns.items())
    return numbered, turns


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[*_`]+", "", s or "")).strip().lower()


def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", _norm(s)))


def check_grounding(triples, turns):
    """Ground each triple's evidence against the cited turn(s) PLUS their
    immediate neighbors (ASR speaker-turn boundaries are unreliable, so a quote
    that stitches across a bad split is a repair, not a hallucination).
      exact    = quote is a literal substring of the cited window
      stitched = >=80% of evidence tokens present in the window (cross-boundary/ASR repair)
      miss     = evidence not supported by the cited turns"""
    maxid = max(turns) if turns else 0
    stats = {"exact": 0, "stitched": 0, "miss": 0}
    for t in triples:
        ev = _norm(t.get("evidence", ""))
        ids = [int(m) for c in (t.get("evidence_turns") or [])
               for m in re.findall(r"\d+", str(c))]
        win = sorted({j for i in ids for j in (i - 1, i, i + 1) if 1 <= j <= maxid})
        wtext = " ".join(turns.get(i, "") for i in win)
        nwin = _norm(wtext)
        if ev and nwin and ev in nwin:
            g = "exact"
        else:
            et, wt = _tokens(ev), _tokens(wtext)
            cov = len(et & wt) / max(1, len(et))
            g = "stitched" if cov >= 0.8 else "miss"
        t["_grounded"] = g
        stats[g] += 1
    return stats


def graph_stats(triples):
    """Every subject and object is a node; every triple is an edge. Node
    salience is emergent from degree (how often it recurs) — no entity/concept
    typing. Returns (n_nodes, degree Counter)."""
    from collections import Counter
    deg = Counter()
    for t in triples:
        deg[t["subject"]] += 1
        deg[t["object"]] += 1
    return len(deg), deg


PROMPT = _ic.fill_program("""You are the fact-extraction stage of a quantitative-memetics pipeline studying the {PROGRAM} ({SPAN_ACTIVE}). Your job is to mine ONE meeting transcript into a compressed, structured representation: a summary plus a set of (subject, relation, object) triples. Downstream, relational triples become edges in an entanglement graph and claim triples are handed to an agent that verifies them against external sources, so precision and grounding matter more than volume.

## Transcript metadata
- Title: {title}
- Date: {date}
- Type: {mtype}
- Named participants: {participants}

The transcript is machine-transcribed (ASR): proper nouns are often garbled, and speaker labels like "Speaker 4" are unreliable. Use the participant list to attribute speech where you are confident; when a name or fact is uncertain due to ASR, still extract it but set confidence to "low" and put your best-guess canonical form in the triple.

Each speaker turn is prefixed with a turn id like [T12]. You will cite these ids to ground every triple.

## Transcript
{transcript}

## Task
Return ONLY JSON (no fences, no prose), of the form:
{{
  "summary": "3-6 sentences capturing what this meeting was about and what changed/was decided.",
  "triples": [
    {{
      "subject": "canonical entity or person (lowercase-ish kebab, e.g. 'ada', 'acme-db', 'harbor-os')",
      "relation": "a short predicate, verb-like (e.g. 'proposes', 'merged-into', 'debugged', 'is-a', 'overweight'). Use natural predicates; do NOT force a fixed vocabulary.",
      "object": "the target entity, concept, or short phrase",
      "subject_type": "a short freeform tag for display only (e.g. person, project, org, tool, idea, quality, event) — NOT used to decide anything",
      "object_type": "same freeform tagging",
      "modality": "asserted | hedged | reported | hypothetical | proposed | planned | negated",
      "certainty": 0.0,
      "extraction_confidence": "high | medium | low",
      "verifiable": true or false,
      "evidence_turns": ["T12", "T13"],
      "evidence": "an EXACT verbatim substring (<=25 words) copied character-for-character from the cited turn(s)"
    }}
  ]
}}

Evidence grounding (strict):
- `evidence_turns` = the turn id(s) (e.g. ["T42"]) whose text supports this triple. A triple may span a few consecutive turns; cite all of them.
- `evidence` = an EXACT substring copied verbatim from those turns — do NOT paraphrase, clean up ASR, or fix grammar. It must be findable by literal string search in the cited turn. If the only supporting words are garbled, quote them garbled.

Guidance:
- EVERY subject and object becomes a NODE, and every triple is an EDGE — there is no entity-vs-concept distinction. A person, a project, an idea, a proposed tool, and a one-off characterization are all just nodes. Their importance is decided later by how often they recur across the corpus, not by any type you assign. So the one thing that matters is naming: give the SAME node the SAME canonical string every time it appears (e.g. always `harbor-os`, never also `HOS` or `harbor os`), so repeated mentions collapse onto one node.
- The `*_type` fields are just display tags; do not agonize over them and never let them change what you extract.
- Set `verifiable` = true whenever an external source (a repo, an article, a public record) could confirm or refute the triple — these get routed to a fact-checking agent.

Two INDEPENDENT certainty axes — keep them separate, they routinely disagree:
- `extraction_confidence` (high/medium/low): how sure you are you READ and ATTRIBUTED this correctly — driven by ASR garble and speaker-label reliability. A clearly-transcribed sentence is "high" even if the claim it makes is dubious.
- `certainty` (0.0-1.0): how likely the claim is actually TRUE IN THE WORLD, judged from HOW it is put forward in the transcript. This is about the claim, not the audio. Calibrate off `modality`:
  * asserted (flat statement of fact) -> ~0.85-0.95
  * reported (secondhand / "X told me") -> ~0.6-0.8
  * hedged ("I think", "maybe", "somehow", "I don't know") -> ~0.25-0.5
  * planned (a decided future intention) -> ~0.5-0.7 (intended, not yet real)
  * proposed (an idea/suggestion to build or do) -> ~0.15-0.35 (not real yet)
  * hypothetical (conditional / "what if" / example-for-illustration) -> ~0.1-0.3
  * negated (stated as NOT the case) -> encode the negation in the triple; certainty is how sure it's false-as-stated
- `modality` names the linguistic frame that drives `certainty`, so a reader/agent can see WHY a fact is uncertain (a hedged guess vs. a mere proposal vs. a plan) rather than just a number.
- Example: "I think Ada somehow got the grant, maybe, I don't know" -> extraction_confidence "high" (clearly said), modality "hedged", certainty ~0.3.

- Prefer load-bearing facts over chatter. Aim for the ~40-70 most significant triples; do not pad with trivia.

CAPTURE NARRATIVE ARCS, not just standalone facts:
- When a speaker RECOUNTS a sequence of events about a person or project — even in passing, even secondhand — emit a triple for EACH step of that story. Example shape: someone drifted away -> was told to step back for a few weeks -> came back with an update. That is three triples (tagged modality "reported" since it is one person's account), not zero.
- Pay special attention to PERIPHERAL subjects: people or projects who are not in the meeting but whose storyline is narrated by someone who is (a coordinator describing another team's trajectory, a withdrawal and later re-engagement, a mediation). These arcs are the easiest to miss and the most valuable — do not skip a subject's storyline just because they are discussed rather than present.
- If a coordinator characterizes several people's working styles or trajectories in one monologue, extract each person's arc separately.

- Ground every triple with an exact-substring quote and its turn id(s).""")


def parse_facts(raw):
    """Parse the model's JSON, salvaging complete triples if the response was
    truncated mid-object (rather than throwing away the whole extraction)."""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    s = clean[clean.find("{"):clean.rfind("}") + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r'"summary"\s*:\s*"(.*?)"\s*,\s*"triples"', s, re.S)
        summary = m.group(1) if m else ""
        triples = []
        for o in re.findall(r"\{[^{}]*\}", s):  # triples are flat (no nested braces)
            try:
                triples.append(json.loads(o))
            except json.JSONDecodeError:
                pass
        print(f"  [salvaged {len(triples)} triples from truncated/invalid JSON]",
              file=sys.stderr)
        return {"summary": summary, "triples": triples}


def lookup_meta(tid):
    """Pull transcript metadata (title/date/type/participants) from the vault
    index.csv by transcript id; fall back to bare id if not found."""
    import csv
    idx = os.path.join(HERE, "data", "vault", "index.csv")
    if os.path.exists(idx):
        for r in csv.reader(open(idx)):
            if r and r[0] == tid:
                return {"title": r[7] if len(r) > 7 and r[7] else tid,
                        "date": r[1] if len(r) > 1 else "",
                        "mtype": r[5] if len(r) > 5 else "",
                        "participants": r[11] if len(r) > 11 else ""}
    return {"title": tid, "date": "", "mtype": "", "participants": ""}


def generate(path, model):
    import anthropic
    with open(path) as f:
        transcript = f.read()
    numbered, turns = number_turns(transcript)
    tid = os.path.splitext(os.path.basename(path))[0]
    meta = lookup_meta(tid)
    prompt = PROMPT.format(transcript=numbered, **meta)
    client = anthropic.Anthropic(max_retries=3, timeout=600)
    is_fable = "fable" in model or "mythos" in model
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if is_fable:
        # Fable: thinking is always on (omit the param); give budget for it + output
        kwargs["max_tokens"] = 32000
    else:
        kwargs["max_tokens"] = 24000
        kwargs["thinking"] = {"type": "disabled"}
    with client.messages.stream(**kwargs) as st:
        msg = st.get_final_message()
    raw = next((b.text for b in msg.content if b.type == "text"), "")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"{tid}.raw.txt"), "w") as f:
        f.write(raw)
    data = parse_facts(raw)
    data["_grounding"] = check_grounding(data["triples"], turns)
    data["_meta"] = meta
    data["_model"] = model
    data["_usage"] = {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens}
    with open(os.path.join(OUTDIR, f"{tid}.json"), "w") as f:
        json.dump(data, f, indent=2)
    return data, tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--model", default="claude-opus-4-8")
    args = ap.parse_args()
    load_env()
    data, tid = generate(args.transcript, args.model)
    u = data["_usage"]
    rate = (10, 50) if ("fable" in args.model or "mythos" in args.model) else (5, 25)
    cost = u["in"] / 1e6 * rate[0] + u["out"] / 1e6 * rate[1]
    print(f"=== {tid} | {args.model} | in={u['in']} out={u['out']} ~${cost:.3f} ===\n")
    print("SUMMARY\n" + data["summary"] + "\n")
    tr = data["triples"]
    n_nodes, deg = graph_stats(tr)
    print(f"GRAPH: {len(tr)} edges · {n_nodes} nodes · "
          f"{sum(v > 1 for v in deg.values())} nodes recur within this transcript · "
          f"{sum(bool(t.get('verifiable')) for t in tr)} verifiable\n")
    g = data["_grounding"]
    gmark = {"exact": "  ", "stitched": "≈ ", "miss": "!!"}
    for t in tr:
        cert = t.get("certainty", 0.0)
        xc = t.get("extraction_confidence", "?")[0].upper()  # H/M/L
        v = " ✓ver" if t.get("verifiable") else ""
        turns_ref = ",".join(str(x) for x in (t.get("evidence_turns") or []))
        print(f"  {gmark[t['_grounded']]} cert={cert:.2f} x{xc} {t.get('modality','?'):<12} "
              f"({t['subject']}) --{t['relation']}--> ({t['object']})  [{turns_ref}]{v}")
    grounded = g["exact"] + g["stitched"]
    print(f"\nGROUNDING: exact={g['exact']} stitched={g['stitched']} miss={g['miss']}"
          f"  ({100*grounded//max(1,len(tr))}% grounded, {100*g['exact']//max(1,len(tr))}% exact)")
    miss = [t for t in tr if t["_grounded"] == "miss"]
    if miss:
        print("  ungrounded (evidence quote not found in transcript):")
        for t in miss[:10]:
            print(f"    ({t['subject']}) --{t['relation']}--> ({t['object']})  ev=\"{t.get('evidence','')[:50]}\"")
    # spotlight: where the two axes disagree (clearly-read but uncertain claims)
    split = [t for t in tr if t.get("extraction_confidence") == "high"
             and t.get("certainty", 1.0) <= 0.4]
    print(f"\nHIGH extraction / LOW certainty (clearly said, but not settled): {len(split)}")
    for t in split[:12]:
        print(f"  cert={t['certainty']:.2f} {t.get('modality'):<12} "
              f"({t['subject']}) --{t['relation']}--> ({t['object']})")
    print(f"\nfull JSON -> data/facts/{tid}.json")


if __name__ == "__main__":
    main()
