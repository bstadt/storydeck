#!/usr/bin/env python3
"""Ingest new transcripts: pull from S3, integrate, then mine them for
  (1) updates to ACTIVE subjects' stories, and
  (2) new POTENTIAL subject suggestions.

Flow:
  sync_transcripts (S3 → data/vault)
    → clean outputs for removed/changed transcripts
    → extract_runner (subjects+beats for new/changed; resumable)
    → corpus rebuild (compile → vocab_merge → langshare → build_showcase)
    → registry rebuild
    → refresh the story of every ACTIVE subject touched by the new transcripts
    → surface new POTENTIAL subjects as suggestions (viewer/data/suggestions.json)

Only ACTIVE subjects get (expensive) story regeneration; new subjects land as
potential and are ranked as suggestions for a human to activate.

Run: python3 ingest.py [--no-story-refresh]
"""
import argparse
import json
import os

import pipeline

HERE = pipeline.HERE
VD = pipeline.VD


def rm(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def outputs_for(tid):
    return [os.path.join(HERE, "data", "subjects", f"{tid}.json"),
            os.path.join(HERE, "data", "beats", f"{tid}.json"),
            os.path.join(HERE, "viewer", "data", f"{tid}.json")]


def canonical_subjects_in(tid):
    canon = json.load(open(os.path.join(HERE, "data", "canonical_vocab.json")))["canon"]
    sp = os.path.join(HERE, "data", "subjects", f"{tid}.json")
    if not os.path.exists(sp):
        return set()
    return {canon.get(s["id"], s["id"]) for s in json.load(open(sp)).get("subjects", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-story-refresh", action="store_true")
    args = ap.parse_args()

    # subjects known before ingest (for new-subject detection)
    prev_ids = set()
    if os.path.exists(pipeline.REGISTRY):
        prev_ids = set(json.load(open(pipeline.REGISTRY))["subjects"])

    out = pipeline.sh("sync_transcripts.py")
    summary = json.loads(out.strip().splitlines()[-1])
    added, changed, removed = summary["added"], summary["changed"], summary["removed"]
    print(f"sync: +{len(added)} / ~{len(changed)} / -{len(removed)}")

    for tid in removed + changed:
        rm(*outputs_for(tid))
    touched = set(added) | set(changed)

    if not touched and not removed:
        print("no transcript changes — nothing to ingest")
        return

    # corpus layer (extract new/changed, then rebuild derived data + registry)
    pipeline.build_corpus(with_extract=True)
    pipeline.build_registry()

    active = pipeline.load_active()

    # (1) refresh the story of ACTIVE subjects touched by the new transcripts
    if not args.no_story_refresh:
        affected = set()
        for tid in touched:
            affected |= canonical_subjects_in(tid) & active
        print(f"\nrefreshing stories for {len(affected)} active subjects touched by new content: {sorted(affected)}")
        for cid in sorted(affected):
            pipeline.sh("gen_narratives.py", "--only", cid)   # reads existing arc as "story so far"
        if affected:
            pipeline.sh("gen_narratives.py", "--compile", quiet=True)

    # (2) surface new POTENTIAL subjects as suggestions
    reg = json.load(open(pipeline.REGISTRY))["subjects"]
    show = json.load(open(os.path.join(VD, "showcase.json")))["subjects"]
    new_ids = [cid for cid in reg if cid not in prev_ids and reg[cid]["status"] == "potential"]
    # rank new subjects by how much of the NEW content they touch
    def new_content_nodes(cid):
        hits = show.get(cid, {}).get("hits", [])
        # map tx index → id
        tx = json.load(open(os.path.join(VD, "showcase.json")))["transcripts"]
        touched_ids = {tx[h[0]]["id"] for h in hits} & touched
        return len(touched_ids)
    suggestions = sorted(
        ({"id": cid, "display": reg[cid]["display"], "type": reg[cid]["type"],
          "n_tx": reg[cid]["n_tx"], "n_nodes": reg[cid]["n_nodes"]}
         for cid in new_ids),
        key=lambda s: (-s["n_nodes"], -s["n_tx"]))
    with open(os.path.join(VD, "suggestions.json"), "w") as f:
        json.dump({"from_transcripts": sorted(touched), "suggestions": suggestions}, f)
    print(f"\n{len(suggestions)} new potential subjects suggested (viewer/data/suggestions.json):")
    for s in suggestions[:12]:
        print(f"  {s['n_nodes']:>3}n {s['n_tx']:>3}tx  {s['id']}  ({s['type']})")

    import csv
    ul = os.path.join(HERE, "data", "usage_log.csv")
    total = sum(float(r["cost_usd"]) for r in csv.DictReader(open(ul))) if os.path.exists(ul) else 0
    print(f"\nDONE. Cumulative API spend: ${total:.2f}")


if __name__ == "__main__":
    main()
