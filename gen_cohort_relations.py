#!/usr/bin/env python3
"""Estimate, per subject, a RANKED list of related cohort subjects with a
per-WEEK strength — a prompt-based read of the same prefill (summaries +
metadata + grounded triples, in temporal order) that narrative generation uses.

For each active subject we hand the model its full temporal prefill plus the
roster of the OTHER cohort subjects, and ask it to rank which of them this
subject is most related to across the program and how strong that tie is in
each meeting-week. The list order IS the rank (strongest first), which the
renderer turns into inverse-rank edge weights. Aggregating the 19 DIRECTED reads
gives a time series of directed cohort graphs (build_cohort_viz.py).

Output: data/facts/cohort_relations.json = {
  subject: { "relations": [ {"entity", "overall", "weeks":{"YYYY-MM-DD":strength},
                             "why"}, ... ] } }   # ranked strongest-first

Usage: python3 gen_cohort_relations.py            # all subjects
       python3 gen_cohort_relations.py brandon    # one subject (debug)
"""
import json
import re
import sys
import instance_config as _ic

import cohort_axis as ax
import extract_facts as ef
import gen_story_triplets as g

MODEL = "claude-opus-4-8"          # structured estimation; Opus is plenty and cheap
WEEKS = ax.weeks()                 # [(key, label, count)] meeting-weeks, chronological
EXCLUDE = _ic.excluded_subjects()   # pseudo-subjects dropped from the graph


def roster_legend(exclude_key):
    """One line per OTHER cohort subject: key plus the name-stems that identify
    it in the text, so the model can map mentions back to canonical keys."""
    lines = []
    for k, spec in g.SUBJECTS.items():
        if k == exclude_key or k in EXCLUDE:
            continue
        aka = ", ".join(spec["stems"])
        lines.append(f"- {k}  (aka: {aka})")
    return "\n".join(lines)


PROMPT = _ic.fill_program("""You are mapping the RELATIONSHIP STRUCTURE of a startup-accelerator cohort ({PROGRAM_SHORT}, {SPAN_FULL}) as it evolves over time. Below is everything mined about ONE subject, **{subject}**: per-meeting summaries and grounded (subject, relation, object) triples with provenance, in STRICT CHRONOLOGICAL ORDER (oldest meeting first). Each meeting block is headed with its date, so you can place ties on the timeline precisely.

{context}

## The other cohort subjects
Estimate {subject}'s relationships to these subjects ONLY (map any name mentions to these canonical keys; ignore people/entities not on this list):
{roster}

## Task
Produce a RANKED list of the cohort subjects that **{subject}** is most related to over the program. Relatedness = genuine connection visible in the data: collaboration, shared work or projects, mentorship/influence, funding, repeated co-occurrence in the same meetings and triples, one shaping the other's trajectory. Rank by overall strength of tie, STRONGEST FIRST — the list order is the ranking and matters. Omit subjects with no real relationship.

For each related subject, give the strength of the tie in the relevant MEETING-WEEKS on this timeline (each entry is `week-start-date: label`):
{weeks}
Use 0.0-1.0 per week (0 = no visible relationship that week, ~0.6 = active working tie, ~0.9 = central/defining relationship that week). CRITICAL — weight weeks by ACTUAL JOINT INTERACTION ONLY: both parties in the same meeting, direct collaboration, a direct hand-off, or one directly shaping the other's work THAT WEEK. Do NOT assign meaningful weight for weeks where one party is merely mentioned, discussed, planned-about, or evaluated in the other's absence — cap such mention-only weeks at 0.1. An expressed DESIRE, intent, or plan to collaborate is NOT interaction — 'X wants to work with Y' scores 0.1 until they actually start working together; do not extend a single real meeting into following weeks unless new joint activity appears in those weeks' blocks. A relationship STARTS the week the two first actually interact, not when one first hears of the other. RESPECT CHRONOLOGY: no weight before first actual contact. Set `overall` to the peak/representative strength.

Output ONLY JSON, no prose:
{{"relations": [
  {{"entity": "<canonical key>", "overall": 0.0-1.0,
    "weeks": {{"2026-06-08": 0.6, "2026-06-15": 0.9}},
    "why": "<one short clause on what the tie is>"}}
]}}
Use the week-start dates (Mondays) shown above as keys; include only weeks with nonzero strength. Rank the `relations` list by `overall` descending.""")


def salvage_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {"relations": []}
    try:
        return json.loads(m.group(0))
    except Exception:
        # trim to the last complete relation object if truncated
        s = m.group(0)
        cut = s.rfind("}}")
        if cut != -1:
            try:
                return json.loads(s[:cut + 2] + "]}")
            except Exception:
                pass
        return {"relations": []}


def estimate(client, key):
    spec = g.SUBJECTS[key]
    context, ntx, ntri, _ = g.build_context(spec)
    week_list = "\n".join(f"- {k}: {lab}" for k, lab, _ in WEEKS)
    prompt = PROMPT.format(subject=key, context=context,
                           roster=roster_legend(key), weeks=week_list)
    with client.messages.stream(model=MODEL, max_tokens=6000,
                                messages=[{"role": "user", "content": prompt}]) as st:
        msg = st.get_final_message()
    txt = next((b.text for b in msg.content if b.type == "text"), "")
    data = salvage_json(txt)
    rels = data.get("relations", [])
    keys = set(g.SUBJECTS)
    rels = [r for r in rels if r.get("entity") in keys and r.get("entity") != key]
    cost = msg.usage.input_tokens / 1e6 * 5 + msg.usage.output_tokens / 1e6 * 25
    print(f"{key}: {ntx} tx · {ntri} tri · {len(rels)} relations · "
          f"in={msg.usage.input_tokens} out={msg.usage.output_tokens} ${cost:.2f}")
    return {"relations": rels}, cost


def main():
    ef.load_env()
    import anthropic
    client = anthropic.Anthropic(max_retries=3, timeout=600)
    only = sys.argv[1:] if len(sys.argv) > 1 else [k for k in g.SUBJECTS if k not in EXCLUDE]
    out, total = {}, 0.0
    for key in only:
        res, cost = estimate(client, key)
        out[key] = res
        total += cost
    dst = "data/facts/cohort_relations.json"
    if len(only) < len(g.SUBJECTS):
        try:
            prev = json.load(open(dst))
        except Exception:
            prev = {}
        prev.update(out)
        out = prev
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\ntotal ${total:.2f} -> {dst} ({len(out)} subjects)")


if __name__ == "__main__":
    main()
