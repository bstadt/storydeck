#!/usr/bin/env python3
"""Storydeck CLI — one entry point over the pipeline stages.

  python3 storydeck.py status                 what exists / what's stale
  python3 storydeck.py ingest <folder>        generic adapter: folder of .md/.txt (+ .json sidecars)
  python3 storydeck.py ingest --coordinationos   CoordinationOS S3 share sync + story refresh
  python3 storydeck.py mine                   facts -> relations -> week groups -> labels  (PAID; confirms first)
  python3 storydeck.py build                  rebuild viewer artifacts (free, local)
  python3 storydeck.py serve                  run the dev server on :8811

Every paid stage prints a cost estimate and asks before spending. All stages
are resumable — already-mined transcripts/weeks are skipped, so re-running
after adding transcripts only pays for what's new.

Instance binding (program framing, subject roster) lives in instance.json;
point STORYDECK_INSTANCE at another file to run a different corpus.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(HERE, script), *args], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"{script} failed ({r.returncode})")


def facts_state():
    """(n_transcripts, n_mined, unmined_ids)"""
    import csv
    ids = []
    idx = os.path.join(HERE, "data", "vault", "index.csv")
    if os.path.exists(idx):
        for r in csv.reader(open(idx)):
            if r and r[0] != "id" and len(r) > 15 and r[15] not in ("removed",):
                ids.append(r[0])
    mined = {os.path.basename(f)[:-5]
             for f in glob.glob(os.path.join(HERE, "data", "facts", "*.json"))}
    unmined = [t for t in ids if t not in mined]
    return len(ids), len(set(ids) & mined), unmined


def status(_):
    n_tx, n_mined, unmined = facts_state()
    print(f"transcripts in vault : {n_tx}")
    print(f"fact-mined           : {n_mined}  ({len(unmined)} pending)")
    for p, label in [("data/facts/week_groups.json", "week groups"),
                     ("data/facts/community_labels.json", "community labels"),
                     ("viewer/geometry.html", "geometry build"),
                     ("viewer/data/showcase.json", "showcase data")]:
        fp = os.path.join(HERE, p)
        print(f"{label:21}: {'present' if os.path.exists(fp) else 'MISSING'}"
              + (f"  ({sum(1 for _ in open(fp)) if p.endswith('.csv') else os.path.getsize(fp)//1024}kB)" if os.path.exists(fp) else ""))
    stories = glob.glob(os.path.join(HERE, "data", "facts", "STORY_*__isaacson.md"))
    print(f"subject stories      : {len(stories)}")


def confirm(msg):
    if os.environ.get("STORYDECK_YES") == "1":
        return True
    try:
        return input(f"{msg} [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def ingest(a):
    if a.coordinationos:
        run("ingest.py")
    elif a.folder:
        run("ingest_local.py", a.folder)
        print("ingested. next: python3 storydeck.py mine")
    else:
        sys.exit("ingest needs <folder> or --coordinationos")


def mine(a):
    n_tx, n_mined, unmined = facts_state()
    if unmined:
        est = len(unmined) * 0.12   # ~$0.12/transcript facts mining (Opus, obs. average)
        if not confirm(f"mine facts for {len(unmined)} transcripts (~${est:.0f})?"):
            return
        run("extract_facts.py")
    else:
        print("facts: up to date")
    if confirm("re-estimate cohort relations (~$0.3-10 depending on novelty)?"):
        run("gen_cohort_relations.py")
    if confirm("regenerate weekly working groups (cheap for new weeks only)?"):
        run("gen_week_groups.py")
    if confirm("re-label communities (~$0.1)?"):
        run("gen_community_labels.py")
    print("mining done. next: python3 storydeck.py build")


def build(_):
    run("build_cohort_viz.py")
    print("geometry rebuilt. (stories/showcase data rebuilds happen inside ingest)")


def serve(_):
    os.execv(sys.executable, [sys.executable, os.path.join(HERE, "serve.py")])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=status)
    pi = sub.add_parser("ingest")
    pi.add_argument("folder", nargs="?", help="folder of .md/.txt transcripts (+ optional .json sidecars)")
    pi.add_argument("--coordinationos", action="store_true", help="sync the CoordinationOS S3 share instead")
    pi.set_defaults(fn=ingest)
    sub.add_parser("mine").set_defaults(fn=mine)
    sub.add_parser("build").set_defaults(fn=build)
    sub.add_parser("serve").set_defaults(fn=serve)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
