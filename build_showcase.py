#!/usr/bin/env python3
"""Export viewer/data/showcase.json for the 3D arborescent/rhizomatic showcase.

Structure:
{
  "transcripts": [                      # ordered by date
    {"id", "date", "title", "mtype", "words",
     "beats": [{"l": label, "k": kind, "s": start, "e": end, "w": words,
                "subj": [canonical ids], "c": [children...]}]}
  ],
  "subjects": {cid: {"display", "type", "n_tx",
                     "hits": [[txIndex, beatPath...], ...]}}   # beatPath = child indices
}
Subjects limited to the langshare set (>=2 transcripts).
"""
import glob
import json
import os
import re

import person_voice

HEADER_RE = re.compile(r"\b(header|frontmatter|preamble|backfill|recon)\b", re.I)
# words that carry no grounding content (header/provenance vocab + connectors)
STOP = set("header headers frontmatter front matter preamble backfill recon vault "
           "metadata meta note notes file doc document transcript recording otter "
           "asr provenance summary keywords keyword dump auto generated and the a of "
           "with to for".split())

def is_header_beat(b):
    """Drop a beat only if it is *dominantly* a file-header/provenance beat — i.e.
    its label matches the header vocabulary AND nothing substantive remains once
    header/connector words are stripped. Keeps e.g. 'Header, recap & next steps'."""
    lab = b.get("label", "")
    low = lab.strip().lower()
    if not (HEADER_RE.search(lab) or low.startswith("vault ")):
        return False
    remainder = [w for w in re.split(r"[^a-z0-9]+", low) if w and w not in STOP and len(w) > 2]
    return len(remainder) == 0

HERE = os.path.dirname(os.path.abspath(__file__))

vocab = json.load(open(os.path.join(HERE, "data", "canonical_vocab.json")))
CANON = vocab["canon"]
LS = json.load(open(os.path.join(HERE, "viewer", "data", "langshare.json")))
KEEP = set(LS["subjects"].keys())

ALIAS_STOP = set("the this that a an their our your his her its these those and or of "
                 "to for in on at is it os app tool map list thing stuff one".split())

def qualifying_aliases(aliases, display):
    """Aliases specific enough to count as grounding — drop generic ones like
    'the map' / 'this OS' / 'the list' that would over-match."""
    out = set()
    for a in list(aliases) + [display]:
        if not a or len(a) < 4:
            continue
        toks = [t for t in re.split(r"[^a-z0-9]+", a.lower()) if t]
        if any(len(t) >= 4 and t not in ALIAS_STOP for t in toks):
            out.add(a.lower())
    return out

def transcript_grounding(d):
    """Per canonical subject: set of mention lines + set of specific alias strings."""
    ment = {}
    alias = {}
    for sub in d["subjects"]:
        cid = CANON.get(sub["id"], sub["id"])
        ment.setdefault(cid, set()).update(m["line"] for m in sub["mentions"])
        alias.setdefault(cid, set()).update(qualifying_aliases(sub.get("aliases", []), sub["display"]))
    return ment, alias

def grounded(cid, s0, e0, lines, ment, alias):
    if any(s0 <= ln <= e0 for ln in ment.get(cid, ())):
        return True
    al = alias.get(cid)
    if not al:
        return False
    span = "\n".join(lines[s0 - 1:e0]).lower()
    return any(re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", span) for t in al)

transcripts = []
hits = {}  # cid -> [[tx_idx, path...]]
tx_docs = {}  # ti -> {lines, speakers, beats_flat:[(path,start,end)]}

files = sorted(glob.glob(os.path.join(HERE, "viewer", "data", "2026*.json")),
               key=lambda p: json.load(open(p))["date"] + json.load(open(p))["id"])


def slim(b, path, ti, lines, ment, alias):
    tagged = {CANON.get(s, s) for s in (b.get("subjects") or [])} & KEEP
    # a subject is a node here only if it is textually grounded in the beat span
    subj = sorted(c for c in tagged if grounded(c, b["start"], b["end"], lines, ment, alias))
    for cid in subj:
        hits.setdefault(cid, []).append([ti] + path)
    kept_children = [c for c in (b.get("children") or []) if not is_header_beat(c)]
    return {"l": b.get("label", "?"), "k": b.get("kind", "other"),
            "s": b["start"], "e": b["end"], "w": b.get("word_count", 0),
            "sm": (b.get("summary") or "")[:400],
            "subj": subj,
            "c": [slim(c, path + [j], ti, lines, ment, alias)
                  for j, c in enumerate(kept_children)]}


dropped_beats = 0
for ti, p in enumerate(files):
    d = json.load(open(p))
    lines = d["lines"]
    ment, alias = transcript_grounding(d)
    kept_top = [b for b in (d.get("beats") or []) if not is_header_beat(b)]
    dropped_beats += len(d.get("beats") or []) - len(kept_top)
    kept = [slim(b, [j], ti, lines, ment, alias) for j, b in enumerate(kept_top)]
    flat = []
    def _walk(bs, pre):
        for j, b in enumerate(bs):
            flat.append((pre + [j], b["s"], b["e"]))
            _walk(b.get("c") or [], pre + [j])
    _walk(kept, [])
    tx_docs[ti] = {"lines": lines, "speakers": d.get("speakers") or [], "flat": flat}
    transcripts.append({
        "id": d["id"], "date": d["date"], "title": d["title"],
        "mtype": d["meeting_type"],
        "words": sum(len(l.split()) for l in d["lines"]),
        "beats": kept,
    })

# --- voice augmentation: attribute what a PERSON said, not just when discussed ---
voice = {}  # cid -> set("ti/path")
voice_added = 0
for cid, m in LS["subjects"].items():
    if m["type"] != "person":
        continue
    disp = m["display"]
    existing = {"/".join(map(str, h)) for h in hits.get(cid, [])}
    vkeys = set()
    for ti, doc in tx_docs.items():
        if not any(doc["speakers"]):
            continue
        for path, sh in person_voice.voice_beats(disp, doc["lines"], doc["speakers"], doc["flat"]):
            key = "/".join(map(str, [ti] + path))
            if key not in existing:
                hits.setdefault(cid, []).append([ti] + path)
                voice_added += 1
            vkeys.add(key)
    if vkeys:
        voice[cid] = vkeys

subjects = {}
for cid, hs in hits.items():
    m = LS["subjects"][cid]
    subjects[cid] = {"display": m["display"], "type": m["type"],
                     "n_tx": len({h[0] for h in hs}), "hits": hs}
    if cid in voice:
        subjects[cid]["voice"] = sorted(voice[cid])

out = {"transcripts": transcripts, "subjects": subjects}
path = os.path.join(HERE, "viewer", "data", "showcase.json")
with open(path, "w") as f:
    json.dump(out, f)
print(f"{len(transcripts)} transcripts, {len(subjects)} subjects, "
      f"{sum(len(h) for h in hits.values())} beat hits (+{voice_added} voice), "
      f"{dropped_beats} header beats dropped -> {path} ({os.path.getsize(path)//1024}KB)")
