#!/usr/bin/env python3
"""Novel (custom) subject support: turn an arbitrary user-typed term into a tracked
subject by searching the corpus for it — no pre-extraction required.

For a term (e.g. "James", "prediction market moat", "Berlin"), we:
  1. grep every transcript's lines for the term (word-boundary, case-insensitive),
  2. map matched lines to the deepest beats that contain them → grounded nodes,
  3. assemble a chronological digest (beat labels/summaries + matched lines as
     quotes) in the same shape build_digests() produces,
so the standard narrative generator can write a grounded arc for it, and the deck
can render it, exactly like an extracted subject.

Custom subjects are recorded in data/custom_subjects.json so a corpus rebuild can
re-derive their (possibly changed) grounded nodes.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
VD = os.path.join(HERE, "viewer", "data")
CUSTOM_FILE = os.path.join(HERE, "data", "custom_subjects.json")


def slugify(term):
    return re.sub(r"[^a-z0-9]+", "-", term.strip().lower()).strip("-") or "custom"


def _node_at(beats, path):
    n = beats[path[0]]
    for i in path[1:]:
        n = n["c"][i]
    return n


def _grounded_paths(beats, matched):
    """Deepest beats whose [s,e] span contains a matched line."""
    out = set()

    def rec(nodes, prefix):
        for i, n in enumerate(nodes):
            path = prefix + [i]
            in_node = [ln for ln in matched if n["s"] <= ln <= n["e"]]
            if not in_node:
                continue
            kids = n.get("c") or []
            covered = set()
            for c in kids:
                for ln in in_node:
                    if c["s"] <= ln <= c["e"]:
                        covered.add(ln)
            if kids and covered:
                rec(kids, path)
            # mark this node if it holds matches not captured by any child (or is a leaf)
            if any(ln not in covered for ln in in_node):
                out.add(tuple(path))
    rec(beats, [])
    return out


def build(term):
    """Return (slug, hits, entries) for a novel term, where hits are showcase-style
    [ti, *path] and entries are build_digests-shaped per-transcript records."""
    show = json.load(open(os.path.join(VD, "showcase.json")))
    TX = show["transcripts"]
    rx = re.compile(r"\b" + re.escape(term.strip()) + r"\b", re.I)
    slug = slugify(term)
    hits, entries = [], []
    for ti, tx in enumerate(TX):
        vp = os.path.join(VD, f"{tx['id']}.json")
        if not os.path.exists(vp):
            continue
        lines = json.load(open(vp)).get("lines", [])
        matched = [n for n, l in enumerate(lines, 1) if l and rx.search(l)]
        if not matched:
            continue
        paths = _grounded_paths(tx["beats"], set(matched))
        if not paths:
            continue
        beats = []
        for p in sorted(paths):
            node = _node_at(tx["beats"], list(p))
            quote = next((lines[ln - 1][:160] for ln in matched if node["s"] <= ln <= node["e"]), "")
            beats.append({"label": node["l"], "summary": (node.get("sm") or "")[:180],
                          "voice": False, "w": node.get("w", 0), "quote": quote})
            hits.append([ti] + list(p))
        beats.sort(key=lambda x: (bool(x["quote"]), x["w"]), reverse=True)
        entries.append({"tx": tx["id"], "date": tx["date"], "title": tx["title"][:80],
                        "mtype": tx["mtype"], "beats": beats[:4]})
    entries.sort(key=lambda e: e["date"])
    return slug, hits, entries


def load_custom():
    return json.load(open(CUSTOM_FILE)) if os.path.exists(CUSTOM_FILE) else {}


def record_custom(slug, term):
    cur = load_custom()
    cur[slug] = term
    json.dump(cur, open(CUSTOM_FILE, "w"))


def inject_into_showcase(slug, term, hits, n_tx):
    """Add/refresh the synthetic subject in showcase.json so the viewer renders it."""
    p = os.path.join(VD, "showcase.json")
    show = json.load(open(p))
    show["subjects"][slug] = {"display": term, "type": "custom", "n_tx": n_tx,
                              "hits": hits, "voice": [], "custom": True}
    json.dump(show, open(p, "w"))


def reapply_all():
    """Re-derive grounded nodes for every recorded custom subject (after a rebuild)."""
    for slug, term in load_custom().items():
        s2, hits, entries = build(term)
        if entries:
            inject_into_showcase(slug, term, hits, len(entries))
