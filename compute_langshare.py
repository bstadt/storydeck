#!/usr/bin/env python3
"""Compute per-subject langshare over time from beat trees + canonical vocab.

Langshare of subject S in a transcript = (words on lines covered by any beat
tagged S, union of ranges — no double counting parent/child) / total words.
Weekly aggregate = sum covered words / sum total words across the week's
transcripts.

Outputs:
  data/langshare_transcripts.csv  (transcript, date, subject, words, total_words)
  data/langshare_weekly.csv       (week, subject, words, total_words, share)
"""
import csv
import datetime
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

vocab = json.load(open(os.path.join(HERE, "data", "canonical_vocab.json")))
CANON, META = vocab["canon"], vocab["meta"]


def walk(beats, out):
    for b in beats:
        for s in b.get("subjects") or []:
            out[CANON.get(s, s)].append((b["start"], b["end"]))
        walk(b.get("children") or [], out)


def covered_words(ranges, word_by_line):
    lines = set()
    for a, b in ranges:
        lines.update(range(a, b + 1))
    return sum(word_by_line.get(n, 0) for n in lines)


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "viewer", "data", "2026*.json"))):
        d = json.load(open(p))
        if not d.get("beats"):
            continue
        word_by_line = {i + 1: len(l.split()) for i, l in enumerate(d["lines"])}
        total = sum(word_by_line.values())
        ranges = defaultdict(list)
        walk(d["beats"], ranges)
        for subj, rs in ranges.items():
            w = covered_words(rs, word_by_line)
            rows.append({"transcript": d["id"], "date": d["date"], "subject": subj,
                         "words": w, "total_words": total})
    with open(os.path.join(HERE, "data", "langshare_transcripts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["transcript", "date", "subject", "words", "total_words"])
        w.writeheader()
        w.writerows(rows)

    # weekly: ISO Monday of each date
    week_total = defaultdict(int)
    week_subj = defaultdict(int)
    seen_tx_week = {}
    for r in rows:
        d = datetime.date.fromisoformat(r["date"])
        wk = (d - datetime.timedelta(days=d.weekday())).isoformat()
        week_subj[(wk, r["subject"])] += r["words"]
        seen_tx_week[r["transcript"]] = (wk, r["total_words"])
    for tx, (wk, tw) in seen_tx_week.items():
        week_total[wk] += tw
    with open(os.path.join(HERE, "data", "langshare_weekly.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["week", "subject", "words", "total_words", "share"])
        for (wk, subj), words in sorted(week_subj.items()):
            tw = week_total[wk]
            w.writerow([wk, subj, words, tw, f"{words / tw:.5f}"])
    n_subj = len(set(r["subject"] for r in rows))
    print(f"langshare: {len(rows)} transcript-subject rows, {n_subj} canonical subjects, "
          f"{len(week_total)} weeks")

    # --- viewer export: subjects appearing in >=2 transcripts, with weekly series
    #     + per-transcript drilldown rows ---------------------------------------
    per_subj = defaultdict(lambda: {"weekly": {}, "tx": [], "words": 0, "n_tx": 0})
    for r in rows:
        e = per_subj[r["subject"]]
        e["tx"].append({"t": r["transcript"], "d": r["date"],
                        "w": r["words"], "share": round(r["words"] / r["total_words"], 4)})
        e["words"] += r["words"]
        e["n_tx"] += 1
    for (wk, subj), words in week_subj.items():
        per_subj[subj]["weekly"][wk] = round(words / week_total[wk], 5)

    titles = {}
    import glob as _g
    for p in _g.glob(os.path.join(HERE, "viewer", "data", "2026*.json")):
        d = json.load(open(p))
        titles[d["id"]] = d["title"]

    out_subjects = {}
    for subj, e in per_subj.items():
        if e["n_tx"] < 2:
            continue
        m = META.get(subj, {})
        e["tx"].sort(key=lambda x: x["d"])
        out_subjects[subj] = {"display": m.get("display", subj), "type": m.get("type", "concept"),
                              "words": e["words"], "n_tx": e["n_tx"],
                              "weekly": e["weekly"], "tx": e["tx"]}
    export = {
        "weeks": sorted(week_total),
        "week_words": {k: week_total[k] for k in week_total},
        "subjects": out_subjects,
        "titles": titles,
        "canon": CANON,
    }
    with open(os.path.join(HERE, "viewer", "data", "langshare.json"), "w") as f:
        json.dump(export, f)
    print(f"viewer export: {len(out_subjects)} subjects (>=2 transcripts) -> viewer/data/langshare.json")


if __name__ == "__main__":
    main()
