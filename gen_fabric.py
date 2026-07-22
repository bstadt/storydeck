#!/usr/bin/env python3
"""FABRIC: weave the corpus into a left-to-right time DAG of entangling
storylines. One Fable call gets the whole mined corpus (per-meeting summaries +
provenance triples, temporally ordered) plus every subject's STRAND, and returns
a DAG: nodes = dated story events tagged with the subjects they involve, edges =
storyline continuation / entanglement (merges, collaborations, splits).

Output: data/facts/fabric.json  {"nodes":[{id,date,label,subjects,desc}],
                                 "edges":[{"from","to"}]}
"""
import glob
import json
import os
import re

import extract_facts as ef

MODEL = "claude-opus-4-8"

def lean_corpus(numbered=False):
    """Chronological summaries + bare triples (no certainty/turn markup) — the
    full CORPUS_GRAPH payload overflows 1M tokens; fabric doesn't need provenance.
    numbered=True heads each block with [#i], numbers each triple line ((j) prefix)
    so answers can cite individual triples as [#i.j], and returns (text, legend)
    where legend = [{i, date, title, id, triples}] for citation rendering."""
    tx = {}
    for jf in glob.glob("data/facts/*.json"):
        tid = os.path.basename(jf)[:-5]
        if tid.endswith((".orig", ".insights")):
            continue
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        if not isinstance(d, dict) or "triples" not in d:
            continue
        m = ef.lookup_meta(tid)
        tx[tid] = (m["date"], m["title"],
                   d.get("summary", ""),
                   [f"{t['subject']} -> {t['relation']} -> {t['object']}"
                    for t in d["triples"]])
    blocks, legend = [], []
    for i, tid in enumerate(sorted(tx, key=lambda t: tx[t][0] or "")):
        date, title, summ, rows = tx[tid]
        if numbered:
            head = f"### [#{i}] [{date}] {title}"
            body = "\n".join(f"({j}) {r}" for j, r in enumerate(rows))
        else:
            head = f"### [{date}] {title}"
            body = "\n".join(rows)
        blocks.append(f"{head}\n{summ}\n{body}")
        legend.append({"i": i, "date": date, "title": title, "id": tid,
                       "triples": rows})
    text = "\n\n".join(blocks)
    return (text, legend) if numbered else text


def main():
    raise SystemExit("the fabric DAG generator was retired with the fabric tab; "
                     "this module now only provides lean_corpus()")


if __name__ == "__main__":
    main()
