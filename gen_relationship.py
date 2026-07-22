#!/usr/bin/env python3
"""Generate a RELATIONSHIP narrative for a pair of subjects: how their two stories
intertwine across the program — converging, diverging, in tension, fusing.

Reuses gen_narratives.build_digests() (showcase-hit-driven, includes person voice)
so each subject's beats come from the same grounded material the solo arcs use.
The pair's shared conversations (both subjects grounded in the same transcript) are
the spine of the relationship; solo conversations are given as context so the model
can see them drifting together or apart.

Output data/relationships/<a>__<b>.json (a,b sorted), then compile into
viewer/data/relationships.json.

Usage:
  python3 gen_relationship.py --pair <cidA> <cidB>
  python3 gen_relationship.py --compile
"""
import argparse
import glob
import json
import os
import re
import time
import instance_config as _ic

import gen_narratives as gn

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-opus-4-8"
OUTDIR = os.path.join(HERE, "data", "relationships")


def pair_key(a, b):
    return "__".join(sorted([a, b]))


PROMPT = _ic.fill_program("""You are writing a RELATIONSHIP synthesis for a quantitative-memetics study of the {PROGRAM} ({SPAN_ACTIVE}). Two subjects have been selected and we want to understand how their stories intertwine over time — whether they grow together or apart, where they collide, align, or diverge, and the narrative of their relationship.

- **A = {da}** ({ta})
- **B = {db}** ({tb})

Below is the chronological record. Conversations where BOTH appear are marked **[BOTH]** — these are the spine of their relationship (each lists A's beats and B's beats). Conversations where only one appears are marked **[A]** or **[B]** — context for whether they are drifting together or apart. Beats marked **[SPOKE]** are ones a person-subject actually spoke in; «quotes» are ASR (proper nouns often garbled).

Shared conversations: {n_both}. {da}-only: {n_a}. {db}-only: {n_b}.

## Record
{record}

## Task
Write, as JSON (no fences, no commentary):
{{
 "trajectory": {{
   "label": "ONE word for the CHARACTER of their bond — one of: entangled | parallel | hierarchical | one-sided | episodic | mutual | oppositional. This is the NATURE of the relationship, NOT whether they co-occur more or less over time (that co-presence trend is measured separately and shown alongside your label).",
   "summary": "one sentence on how their relationship moved over Mar→Jul 2026 (e.g. 'started as separate threads, fused into one product conversation in May, then drifted apart as one receded')."
 }},
 "arc_beats": [
   {{"text": "One sentence (or two short ones) of the relationship's long arc — a moment or movement in how A and B relate.", "tx": ["<transcript id>", ...]}},
   ...
 ],
 "per_tx": {{
   "<shared transcript id>": "1-2 sentences: what THIS joint conversation revealed about their relationship — alignment, tension, hand-off, a turning point.",
   ...
 }}
}}

`arc_beats` rules: 5-9 beats that, read in order, tell the story of the RELATIONSHIP between A and B (not either one alone) — where their threads first touched, how the entanglement evolved, turning points, where it ended up. Concrete, naming conversations/moments, written for a reader browsing a visualization. Each beat's `tx` lists the transcript ids (verbatim from the record) that ground THAT beat — prefer [BOTH] conversations; use a solo conversation's id only when that solo moment is itself part of the relationship story (e.g. one going quiet while the other surges). Use [] only for pure connective tissue. `per_tx` should cover the shared ([BOTH]) conversations.""")


def build_record(a, b, digests):
    A = {e["tx"]: e for e in digests.get(a, [])}
    B = {e["tx"]: e for e in digests.get(b, [])}
    all_tx = sorted(set(A) | set(B), key=lambda t: (A.get(t) or B.get(t))["date"])
    both = [t for t in all_tx if t in A and t in B]
    a_only = [t for t in all_tx if t in A and t not in B]
    b_only = [t for t in all_tx if t in B and t not in A]

    def beats_str(entry, spoke_ok=True):
        parts = []
        for bt in entry["beats"]:
            tag = "[SPOKE] " if (spoke_ok and bt.get("voice")) else ""
            seg = tag + bt["label"] + (f" — {bt['summary']}" if bt["summary"] else "")
            if bt.get("quote"):
                seg += f'  «{bt["quote"]}»'
            parts.append(seg)
        return "\n    ".join(parts)

    lines = []
    for t in all_tx:
        e = A.get(t) or B.get(t)
        if t in A and t in B:
            lines.append(f"[BOTH] [{e['date']}] {e['title']} ({e['mtype']}) :: id={t}"
                         f"\n  {da_label} beats:\n    {beats_str(A[t])}"
                         f"\n  {db_label} beats:\n    {beats_str(B[t])}")
        elif t in A:
            lines.append(f"[A] [{e['date']}] {e['title']} :: id={t}\n    {beats_str(A[t])}")
        else:
            lines.append(f"[B] [{e['date']}] {e['title']} :: id={t}\n    {beats_str(B[t])}")
    return "\n".join(lines), both, a_only, b_only


# module-scope labels used by build_record (set in generate)
da_label = "A"
db_label = "B"


def generate(a, b):
    global da_label, db_label
    import anthropic
    with open(os.path.join(HERE, ".env")) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)

    digests, meta = gn.build_digests()
    if a not in digests or b not in digests:
        missing = [x for x in (a, b) if x not in digests]
        raise RuntimeError(f"subject(s) not in corpus: {missing}")
    da = meta.get(a, {}).get("display", a)
    db = meta.get(b, {}).get("display", b)
    ta = meta.get(a, {}).get("type", "concept")
    tb = meta.get(b, {}).get("type", "concept")
    da_label, db_label = da, db
    record, both, a_only, b_only = build_record(a, b, digests)

    prompt = PROMPT.format(da=da, db=db, ta=ta, tb=tb, record=record,
                           n_both=len(both), n_a=len(a_only), n_b=len(b_only))
    client = anthropic.Anthropic(max_retries=3)
    for attempt in range(4):
        try:
            with client.messages.stream(model=MODEL, max_tokens=48000,
                                        thinking={"type": "disabled"},
                                        messages=[{"role": "user", "content": prompt}]) as st:
                msg = st.get_final_message()
            gn.log_usage(pair_key(a, b), msg.usage)
            text = next((bl.text for bl in msg.content if bl.type == "text"), "")
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            data = json.loads(text[text.find("{"):text.rfind("}") + 1])
            assert "arc_beats" in data and "trajectory" in data
            data["pair"] = sorted([a, b])
            data["shared_tx"] = both
            # recent-grounding safety net: make the arc's final beat reach the
            # latest SHARED conversations so hovering the arc never dead-ends
            # before the newest joint data (mirrors gen_narratives).
            if both and data["arc_beats"]:
                grounded = {t for bt in data["arc_beats"] for t in bt.get("tx", [])}
                # the ~10 MOST RECENT ungrounded shared convos (both is date-asc)
                recent = set([t for t in reversed(both) if t not in grounded][:10])
                lb = data["arc_beats"][-1]
                lb.setdefault("tx", [])
                for t in both:  # append in ascending date order
                    if t in recent and t not in lb["tx"]:
                        lb["tx"].append(t)
            os.makedirs(OUTDIR, exist_ok=True)
            with open(os.path.join(OUTDIR, f"{pair_key(a, b)}.json"), "w") as f:
                json.dump(data, f)
            return data
        except anthropic.RateLimitError:
            time.sleep(min(60, 5 * 2 ** attempt))
        except (json.JSONDecodeError, AssertionError):
            if attempt >= 2:
                raise
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(10)
            else:
                raise
    raise RuntimeError(f"{pair_key(a, b)}: exhausted retries")


def compile_out():
    out = {}
    for p in glob.glob(os.path.join(OUTDIR, "*.json")):
        out[os.path.basename(p)[:-5]] = json.load(open(p))
    dst = os.path.join(HERE, "viewer", "data", "relationships.json")
    with open(dst, "w") as f:
        json.dump(out, f)
    print(f"compiled {len(out)} relationships -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    if args.compile:
        compile_out()
        return
    if args.pair:
        a, b = args.pair
        print(f"generating relationship {pair_key(a, b)} …")
        data = generate(a, b)
        print(f"  trajectory: {data['trajectory']['label']} — {data['trajectory']['summary']}")
        print(f"  {len(data['arc_beats'])} arc beats, {len(data['shared_tx'])} shared conversations")
        compile_out()


if __name__ == "__main__":
    main()
