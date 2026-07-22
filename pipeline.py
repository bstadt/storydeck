#!/usr/bin/env python3
"""Reproducible fbx-memetics pipeline: corpus → all extracted elements + a
per-subject activation path.

Layers
------
Corpus (subject-agnostic, once per transcript):
  extract_runner  → data/subjects/<id>.json, data/beats/<id>.json
  compile_viewer  → viewer/data/<id>.json, manifest.json
  vocab_merge     → data/canonical_vocab.json
  compute_langshare → viewer/data/langshare.json
  build_showcase  → viewer/data/showcase.json      (grounded nodes for EVERY subject)

Subject registry (active | potential):
  build_registry  → viewer/data/registry.json

Per-subject activation (the reproducible "subject → story" pipeline):
  activate(id)    → data/narratives/<id>.json (grounded arc_beats) + status=active
                    → viewer/data/narratives.json

Only subjects with a grounded arc narrative are ACTIVE and power the story UI.
Everything else that appears in ≥2 transcripts is POTENTIAL.

CLI:
  python3 pipeline.py corpus                # rebuild corpus layer (deterministic + vocab merge)
  python3 pipeline.py registry              # rebuild the registry
  python3 pipeline.py activate <subject>    # activate a subject (or fuzzy-matched id)
  python3 pipeline.py list [potential|active]
"""
import csv
import difflib
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VD = os.path.join(HERE, "viewer", "data")
REGISTRY = os.path.join(VD, "registry.json")
COLORS_FILE = os.path.join(HERE, "data", "subject_colors.json")
# distinct, high-contrast palette for active subjects (assigned on activation)
PALETTE = ["#4d94e8", "#2fd39a", "#f0b429", "#ef6a68", "#8b7ce8", "#f291b9",
           "#f08a56", "#3fae3f", "#5ac8e0", "#c98bff", "#e0d24d", "#ff8fb0",
           "#7ee081", "#e86a4d", "#9ab0ff", "#d0d0c8"]


DEACT_FILE = os.path.join(HERE, "data", "deactivated.json")


def load_deactivated():
    return set(json.load(open(DEACT_FILE))) if os.path.exists(DEACT_FILE) else set()


def save_deactivated(s):
    json.dump(sorted(s), open(DEACT_FILE, "w"))


def load_colors():
    return json.load(open(COLORS_FILE)) if os.path.exists(COLORS_FILE) else {}


def assign_color(cid, colors):
    if cid in colors:
        return colors[cid]
    used = set(colors.values())
    col = next((c for c in PALETTE if c not in used), None) or PALETTE[len(colors) % len(PALETTE)]
    colors[cid] = col
    return col


def sh(script, *args, quiet=False):
    r = subprocess.run([sys.executable, os.path.join(HERE, script), *args],
                       cwd=HERE, capture_output=True, text=True,
                       env={**os.environ, "FBX_VAULT": os.path.join(HERE, "data", "vault")})
    if not quiet and r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip(), file=sys.stderr)
        raise SystemExit(f"{script} failed ({r.returncode})")
    return r.stdout


# ---------------- corpus layer ----------------

def build_corpus(with_extract=True):
    if with_extract:
        sh("extract_runner.py")
    sh("compile_viewer_data.py")
    sh("vocab_merge.py")
    sh("compute_langshare.py")
    sh("build_showcase.py")
    # re-derive grounded nodes for any custom (novel) subjects the user activated
    import novel_subject
    novel_subject.reapply_all()


# ---------------- registry ----------------

def has_grounded_narrative(cid):
    p = os.path.join(HERE, "data", "narratives", f"{cid}.json")
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
        return isinstance(d.get("arc_beats"), list) and len(d["arc_beats"]) > 0
    except Exception:
        return False


# geometry/story-file key → canonical viewer key (showcase/registry/stories.json).
# subject_aliases (instance.json) maps a legacy geometry/story key to the viewer subject key.
import instance_config as _ic
SUBJECT_ALIASES = _ic.subject_aliases()


def canonical_cid(cid):
    return SUBJECT_ALIASES.get(cid, cid)


def storied_subjects():
    """Canonical ids that have a triplet-pipeline story: a data/facts/
    STORY_<cid>__isaacson.md on disk or a key in viewer/data/stories.json.
    These subjects (the accelerator-coverage expansion, activated via
    activate_expansion.py) have no old-style data/narratives arc file, so
    build_registry must count this evidence as active too — otherwise a
    registry rebuild silently reverts them all to 'potential'."""
    cids = set()
    for p in glob.glob(os.path.join(HERE, "data", "facts", "STORY_*__isaacson.md")):
        cids.add(os.path.basename(p)[len("STORY_"):-len("__isaacson.md")])
    try:
        cids.update(json.load(open(os.path.join(VD, "stories.json"))))
    except Exception:
        pass
    return {canonical_cid(c) for c in cids}


def _heal_showcase_storied(show, storied):
    """Ensure every storied subject has a showcase entry. build_showcase.py
    regenerates showcase.json from the old canonical vocab, which never saw the
    expansion subjects, so a corpus rebuild would drop them. Display/type come
    from activate_expansion.NEW; transcript counts from the triple corpus
    (same synthesis activate_expansion.py used). Returns True if entries added."""
    subs = show["subjects"]
    missing = sorted(c for c in storied if c not in subs)
    if not missing:
        return False
    import activate_expansion as ax
    import gen_story_triplets as g
    try:
        trip = json.load(open(os.path.join(VD, "triples.json")))
    except Exception:
        trip = {}
    for cid in missing:
        disp, typ = ax.NEW.get(cid, (cid.replace("-", " ").title(), "person"))
        # alias-aware spec lookup: g.SUBJECTS is keyed by the story-file key
        skey = next((k for k, v in SUBJECT_ALIASES.items() if v == cid), cid)
        spec = g.SUBJECTS.get(cid) or g.SUBJECTS.get(skey)
        n_tx = 0
        if spec:
            n_tx = sum(1 for rows in trip.values()
                       if any(g.matches(r[0], spec) or g.matches(r[2], spec) for r in rows))
        subs[cid] = {"display": disp, "type": typ, "n_tx": n_tx, "hits": [], "voice": []}
        print(f"  showcase healed: {cid} ({typ}, n_tx={n_tx})")
    return True


def reconcile_narratives(subs):
    """vocab_merge can rename a subject's canonical slug on re-run (two spellings of one name),
    which orphans its narrative file and silently drops it from active. Rename any
    grounded narrative whose id is no longer a canonical subject to its current
    canonical id. Returns True if anything changed."""
    try:
        canon = json.load(open(os.path.join(HERE, "data", "canonical_vocab.json")))["canon"]
    except Exception:
        return False
    ndir = os.path.join(HERE, "data", "narratives")
    changed = False
    for p in glob.glob(os.path.join(ndir, "*.json")):
        old = os.path.basename(p)[:-5]
        if old in subs:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not (isinstance(d.get("arc_beats"), list) and d["arc_beats"]):
            continue
        new = canon.get(old, old)
        if new in subs and new != old and not os.path.exists(os.path.join(ndir, f"{new}.json")):
            os.rename(p, os.path.join(ndir, f"{new}.json"))
            changed = True
    return changed


def build_registry():
    show = json.load(open(os.path.join(VD, "showcase.json")))
    subs = show["subjects"]
    if reconcile_narratives(subs):        # heal orphaned narratives before counting active
        sh("gen_narratives.py", "--compile", quiet=True)
    storied = storied_subjects()          # triplet-story subjects count as active too
    if _heal_showcase_storied(show, storied):
        with open(os.path.join(VD, "showcase.json"), "w") as f:
            json.dump(show, f)
    colors = load_colors()
    deact = load_deactivated()   # user-hidden subjects: keep their narrative but mark potential
    reg = {}
    for cid, s in subs.items():
        active = (has_grounded_narrative(cid) or cid in storied) and cid not in deact
        entry = {
            "display": s["display"], "type": s["type"],
            "n_tx": s["n_tx"], "n_nodes": len(s["hits"]),
            "status": "active" if active else "potential",
        }
        if active:
            entry["color"] = assign_color(cid, colors)
        reg[cid] = entry
    with open(COLORS_FILE, "w") as f:
        json.dump(colors, f)
    out = {"subjects": reg,
           "n_active": sum(1 for v in reg.values() if v["status"] == "active"),
           "n_potential": sum(1 for v in reg.values() if v["status"] == "potential")}
    with open(REGISTRY, "w") as f:
        json.dump(out, f)
    print(f"registry: {out['n_active']} active, {out['n_potential']} potential")
    # refresh the group-geometry entanglement graph (depends on the active set)
    try:
        sh("compute_entanglement.py", quiet=True)
    except SystemExit:
        pass
    return out


# ---------------- subject resolution ----------------

def resolve_subject(query):
    """Map a user query to a canonical subject id. Exact id → itself; else fuzzy
    match against ids, display names, and aliases. Returns (id, how) or (None, msg)."""
    show = json.load(open(os.path.join(VD, "showcase.json")))
    subs = show["subjects"]
    q = query.strip().lower()
    if query in subs:
        return query, "exact-id"
    # display exact
    for cid, s in subs.items():
        if s["display"].lower() == q:
            return cid, "display"
    # alias exact (from langshare/canon reverse)
    canon = json.load(open(os.path.join(VD, "langshare.json"))).get("canon", {})
    for variant, cid in canon.items():
        if variant.lower() == q and cid in subs:
            return cid, f"alias:{variant}"
    # fuzzy over ids + displays
    names = {cid: s["display"] for cid, s in subs.items()}
    pool = list(names) + [v.lower() for v in names.values()]
    best = difflib.get_close_matches(q, pool, n=1, cutoff=0.6)
    if best:
        m = best[0]
        cid = m if m in names else next(c for c, dn in names.items() if dn.lower() == m)
        return cid, f"fuzzy→{names[cid]}"
    return None, f"no subject matched '{query}' (pick from potential list, or it may be novel — novel-subject search not yet supported)"


# ---------------- activation ----------------

def _unhide(cid):
    """If a subject was soft-removed but still has its narrative, just un-hide it
    (free, no regeneration). Returns True if it handled the (re)activation."""
    deact = load_deactivated()
    if cid in deact:
        deact.discard(cid)
        save_deactivated(deact)
    if has_grounded_narrative(cid):
        build_registry()
        return True
    return False


def deactivate(query):
    cid, how = resolve_subject(query)
    if not cid:
        raise SystemExit(how)
    deact = load_deactivated()
    deact.add(cid)
    save_deactivated(deact)
    build_registry()
    print(f"✓ {cid} hidden (narrative kept)")
    return cid


def activate(query):
    cid, how = resolve_subject(query)
    if not cid:
        raise SystemExit(how)
    if _unhide(cid):                    # was hidden with a narrative → free re-show
        print(f"✓ {cid} un-hidden")
        return cid
    print(f"activating '{query}' → {cid} ({how})")
    # reproducible per-subject step: generate the grounded arc narrative
    npath = os.path.join(HERE, "data", "narratives", f"{cid}.json")
    if not has_grounded_narrative(cid):
        if os.path.exists(npath):
            os.remove(npath)  # drop any stale unstructured narrative
        sh("gen_narratives.py", "--only", cid)
    sh("gen_narratives.py", "--compile", quiet=True)
    build_registry()
    print(f"✓ {cid} is now ACTIVE")
    return cid


def activate_novel(query):
    """Resurface an arbitrary typed term as a tracked subject: grep the corpus,
    build grounded nodes, generate a narrative, and mark it active."""
    import novel_subject as ns
    import gen_narratives as gn
    # if the term already IS an extracted subject, activate that (richer) instead
    slug0 = ns.slugify(query)
    subs = json.load(open(os.path.join(VD, "showcase.json")))["subjects"]
    if slug0 in subs and not subs[slug0].get("custom"):
        print(f"'{query}' is an existing subject ({slug0}); activating it")
        return activate(slug0)
    if _unhide(slug0):                  # a previously-removed custom subject → free re-show
        print(f"✓ {slug0} un-hidden")
        return slug0
    slug, hits, entries = ns.build(query)
    if not entries:
        raise SystemExit(f"'{query}' not found in the corpus")
    print(f"novel subject '{query}' → {slug}: {len(hits)} grounded beats across {len(entries)} conversations")
    ns.record_custom(slug, query)
    ns.inject_into_showcase(slug, query, hits, len(entries))
    # generate the grounded arc (reuses the standard per-subject narrative path)
    with open(os.path.join(HERE, ".env")) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)
    import anthropic
    client = anthropic.Anthropic(max_retries=3)
    os.makedirs(gn.OUTDIR, exist_ok=True)
    gn.gen_one(client, slug, entries, {slug: {"display": query, "type": "custom"}})
    gn.compile_out()
    build_registry()
    print(f"✓ {slug} is now ACTIVE")
    return slug


def compute_relationship(a, b, force=False):
    """Resolve two subjects and generate (or reuse) their relationship narrative.
    Cached at data/relationships/<a>__<b>.json like an activated subject's arc."""
    import gen_relationship as gr
    ca, ha = resolve_subject(a)
    if not ca:
        raise SystemExit(ha)
    cb, hb = resolve_subject(b)
    if not cb:
        raise SystemExit(hb)
    if ca == cb:
        raise SystemExit("pick two different subjects")
    key = gr.pair_key(ca, cb)
    path = os.path.join(HERE, "data", "relationships", f"{key}.json")
    if force or not os.path.exists(path):
        print(f"inferring relationship {ca} ↔ {cb} …")
        gr.generate(ca, cb)
    gr.compile_out()
    return key


def list_subjects(which=None):
    reg = json.load(open(REGISTRY))["subjects"]
    rows = sorted(reg.items(), key=lambda kv: -kv[1]["n_nodes"])
    for cid, s in rows:
        if which and s["status"] != which:
            continue
        print(f"  {s['status']:9} {s['n_nodes']:>3}n {s['n_tx']:>3}tx  {cid}  ({s['type']})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "corpus":
        build_corpus()
        build_registry()
    elif cmd == "registry":
        build_registry()
    elif cmd == "activate":
        activate(sys.argv[2])
    elif cmd == "relationship":
        compute_relationship(sys.argv[2], sys.argv[3], force="--force" in sys.argv)
    elif cmd == "novel":
        activate_novel(" ".join(sys.argv[2:]))
    elif cmd == "deactivate":
        deactivate(sys.argv[2])
    elif cmd == "list":
        list_subjects(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
