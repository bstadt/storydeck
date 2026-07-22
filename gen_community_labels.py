#!/usr/bin/env python3
"""Label the geometry tab's detected communities: one Opus call reads each
stable weekly community's membership-over-time + the relation "why" clauses
among its members and returns a short label + a 1-2 sentence description of
what the group is doing / why they're coordinating.

Reads  data/facts/communities_computed.json  (written by build_cohort_viz.py)
Writes data/facts/community_labels.json      {cid: {"label", "desc"}}
"""
import json
import re
import instance_config as _ic

import extract_facts as ef

PROMPT = _ic.fill_program("""These are communities detected in the collaboration graph of the {PROGRAM} ({SPAN_ACTIVE}), from LLM-estimated working ties between cohort subjects (people and project teams). For each community you get: which members it had in which meeting-weeks, and clauses explaining individual ties between its members.

{blocks}

For EACH community id, infer what this group actually is: a short LABEL (2-5 words, like a caption on a map) and a DESC (1-2 sentences: what the group is doing together / why they're coordinating, grounded in the tie clauses; note evolution if membership shifts). Do not just list the members back. IMPORTANT: members are listed MOST-PERSISTENT-FIRST — name the community for its persistent members and their joint activity, NOT for a well-known figure who joined late or merely mentors it (e.g. a cluster whose durable members are builders coordinating on demo-day work should be named for that work, even if a coordinator's ties are numerous).

Output ONLY JSON: {{"<id>": {{"label": "...", "desc": "..."}}, ...}} covering every id.""")


def main():
    ef.load_env()
    import anthropic
    comm = json.load(open("data/facts/communities_computed.json"))
    blocks = []
    for cid in sorted(comm, key=int):
        info = comm[cid]
        weeks = list(info["frames"].items())
        span = f"{weeks[0][0]} → {weeks[-1][0]}" if weeks else "?"
        evol = "; ".join(f"{w}: {', '.join(m)}" for w, m in weeks[:: max(1, len(weeks) // 5)][:5])
        blocks.append(f"## Community {cid}  (active {span})\nmembership over time: {evol}\n"
                      "ties among members:\n" + "\n".join(f"- {w}" for w in info["whys"]))
    prompt = PROMPT.format(blocks="\n\n".join(blocks))
    client = anthropic.Anthropic(max_retries=3, timeout=300)
    with client.messages.stream(model="claude-opus-4-8", max_tokens=4000,
                                thinking={"type": "disabled"},
                                messages=[{"role": "user", "content": prompt}]) as st:
        msg = st.get_final_message()
    txt = next((b.text for b in msg.content if b.type == "text"), "")
    data = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    json.dump(data, open("data/facts/community_labels.json", "w"), indent=1)
    cost = msg.usage.input_tokens / 1e6 * 5 + msg.usage.output_tokens / 1e6 * 25
    print(f"labeled {len(data)} communities · ${cost:.2f}")
    for cid in sorted(data, key=int):
        print(f"  [{cid}] {data[cid]['label']}")


if __name__ == "__main__":
    main()
