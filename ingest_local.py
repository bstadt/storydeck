#!/usr/bin/env python3
"""Generic ingest adapter: a plain folder of transcripts -> the canonical vault
layout the pipeline reads (data/vault/index.csv + data/vault/transcripts/).

This is the open-source input contract. To run Storydeck on your own corpus you
need only:

  my-vault/
    2026-01-05-team-sync.md        # one file per transcript (.md or .txt);
    2026-01-05-team-sync.json      # optional sidecar metadata (same basename)
    ...

Sidecar JSON fields (all optional): {"date": "YYYY-MM-DD", "title": str,
"meeting_type": str, "participants": "Name; Name"}. Without a sidecar, the
date is parsed from a leading YYYY-MM-DD in the filename and the title from
the rest of it.

Usage: python3 ingest_local.py <src-folder>
Idempotent: re-running updates changed files and reports added/changed/removed
ids as JSON (same contract sync_transcripts.py prints for CoordinationOS).
"""
import csv
import hashlib
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAULT = os.path.join(HERE, "data", "vault")

# the canonical index.csv schema (CoordinationOS-compatible); the pipeline
# reads columns: 0=id, 1=date, 5=meeting_type, 7=title, 11=participants
COLS = ["id", "date", "meeting_time", "team", "teams_covered", "meeting_type",
        "audience", "title", "source_person", "recording_app", "speaker_labels",
        "participants", "fidelity", "derived_from", "other_copies", "status",
        "path", "source_revision", "source_ref", "ca"]


def meta_for(src, base):
    side = os.path.join(src, base + ".json")
    meta = {}
    if os.path.exists(side):
        try:
            meta = json.load(open(side))
        except Exception as e:
            sys.exit(f"bad sidecar {side}: {e}")
    m = re.match(r"(\d{4}-\d{2}-\d{2})[-_ ]*(.*)", base)
    meta.setdefault("date", m.group(1) if m else "")
    meta.setdefault("title", (m.group(2) if m else base).replace("-", " ").replace("_", " ").strip() or base)
    meta.setdefault("meeting_type", "meeting")
    meta.setdefault("participants", "")
    return meta


def main():
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        sys.exit(__doc__)
    src = sys.argv[1]
    os.makedirs(os.path.join(VAULT, "transcripts"), exist_ok=True)
    idx_path = os.path.join(VAULT, "index.csv")
    old = {}
    if os.path.exists(idx_path):
        for r in csv.reader(open(idx_path)):
            if r and r[0] != "id":
                old[r[0]] = r
    # refuse to clobber a vault managed by another adapter (e.g. the
    # CoordinationOS sync): its rows would be dropped from the rewritten index
    foreign = [tid for tid, r in old.items()
               if len(r) > 16 and r[16] and not re.fullmatch(r"transcripts/[^/]+\.md", r[16])]
    if foreign and os.environ.get("STORYDECK_FORCE_INGEST") != "1":
        sys.exit(f"index.csv holds {len(foreign)} rows from another ingest adapter "
                 f"(e.g. {foreign[0]}); running would drop them. "
                 "Set STORYDECK_FORCE_INGEST=1 to override.")
    rows, added, changed = {}, [], []
    for fn in sorted(os.listdir(src)):
        if not fn.endswith((".md", ".txt")):
            continue
        base = os.path.splitext(fn)[0]
        meta = meta_for(src, base)
        dst_rel = os.path.join("transcripts", base + ".md")
        dst = os.path.join(VAULT, dst_rel)
        body = open(os.path.join(src, fn)).read()
        prev = open(dst).read() if os.path.exists(dst) else None
        if prev is None:
            added.append(base)
        elif prev != body:
            changed.append(base)
        if prev != body:
            open(dst, "w").write(body)
        row = [""] * len(COLS)
        row[0] = base
        row[1] = meta["date"]
        row[5] = meta["meeting_type"]
        row[7] = meta["title"]
        row[11] = meta["participants"]
        row[16] = dst_rel
        rows[base] = row
    removed = [tid for tid in old if tid not in rows]
    for tid in removed:
        p = os.path.join(VAULT, "transcripts", tid + ".md")
        if os.path.exists(p):
            os.remove(p)
    with open(idx_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for tid in sorted(rows):
            w.writerow(rows[tid])
    print(json.dumps({"added": added, "changed": changed, "removed": removed}))


if __name__ == "__main__":
    main()
