#!/usr/bin/env python3
"""Compile agent extractions + raw transcripts into viewer-ready JSON.

Inputs:
  data/subjects/<id>.json  — v1 subject/mention extractions (one agent per transcript)
  data/beats/<id>.json     — v2 narrative-beat trees (optional per transcript)
  vault index.csv + transcript files

Outputs:
  viewer/data/<id>.json    — lines + speakers + verified mentions + beat tree
  viewer/data/manifest.json

Verification:
  - mention quotes must be verbatim substrings of the cited line (±3 relocation)
  - beat trees are checked for tiling (coverage/overlap) at each level; violations
    are recorded as warnings, beats are clamped to file range
  - per-line speaker attribution is deterministic where the format allows:
      Otter:   "Name  MM:SS" header lines assign following lines
      inline:  "Speaker N [M:SS]:" inline prefixes
    tracked person = per instance.json program.tracked_person.match
  - per-beat: word_count + tina_share (deterministic, null if no speaker data)
"""
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
VAULT = os.environ.get("FBX_VAULT", os.path.join(HERE, "data", "vault"))
OUT = os.path.join(HERE, "viewer", "data")

OTTER_HDR = re.compile(r"^(.{1,40}?)\s+(\d+:\d\d(?::\d\d)?)\s*$")
INLINE_HDR = re.compile(r"^(Speaker \d+)\s*\[\d+:\d\d(?::\d\d)?\]:\s*")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_excluded():
    p = os.path.join(HERE, "data", "dedup_excluded.json")
    return set(json.load(open(p))) if os.path.exists(p) else set()


def load_index():
    with open(os.path.join(VAULT, "index.csv")) as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def attribute_speakers(lines):
    """Return (speakers, mode): speakers[i] = name or None for line i+1."""
    n_otter = sum(1 for l in lines if OTTER_HDR.match(l.strip()) and len(l.split()) <= 6)
    n_inline = sum(1 for l in lines if INLINE_HDR.match(l))
    speakers = [None] * len(lines)
    if n_inline >= 5:
        for i, l in enumerate(lines):
            m = INLINE_HDR.match(l)
            if m:
                speakers[i] = m.group(1)
        return speakers, "speaker-inline"
    if n_otter >= 5:
        cur = None
        for i, l in enumerate(lines):
            s = l.strip()
            m = OTTER_HDR.match(s)
            if m and len(s.split()) <= 6:
                cur = m.group(1)
                speakers[i] = cur  # header line itself
            else:
                speakers[i] = cur if s else None
        return speakers, "otter-headers"
    return speakers, "none"


GENERIC_SPK = re.compile(r"^Speaker \d+$")


def is_tina(name):
    """The instance's tracked person (schema keys keep the historical 'tina' name)."""
    import instance_config as _ic
    stems = _ic.program().get("tracked_person", {}).get("match", [])
    return bool(name) and any(st in name.lower() for st in stems)


def has_named_labels(speakers):
    """True if any attributed speaker is a real name (not 'Speaker N')."""
    return any(s and not GENERIC_SPK.match(s) for s in speakers)


def verify_mentions(subjects, lines):
    kept = dropped = 0
    nlines = [norm(l) for l in lines]
    for subj in subjects:
        good = []
        for m in subj.get("mentions", []):
            q = norm(m.get("quote", ""))
            ln = m.get("line", 0)
            if not q:
                dropped += 1
                continue
            found = None
            for cand in [ln] + [ln + d for d in (-1, 1, -2, 2, -3, 3)]:
                if 1 <= cand <= len(nlines) and q in nlines[cand - 1]:
                    found = cand
                    break
            if found is None:
                dropped += 1
                continue
            m["line"] = found
            good.append(m)
            kept += 1
        subj["mentions"] = good
        d = subj.get("definition")
        if d and not (1 <= d.get("line", 0) <= len(nlines) and norm(d.get("quote", "")) in nlines[d["line"] - 1]):
            reloc = next((m for m in subj["mentions"] if m.get("kind") == "definition"), None)
            subj["definition"] = {"line": reloc["line"], "quote": reloc["quote"]} if reloc else None
    return kept, dropped


def check_and_enrich_beats(beats, lines, speakers, warnings, lo=1, hi=None, depth=0):
    """Validate tiling, clamp ranges, add word_count/tina_share. Recursive."""
    if hi is None:
        hi = len(lines)
    prev_end = lo - 1
    for b in beats:
        s, e = int(b.get("start", 0)), int(b.get("end", 0))
        if s < lo or e > hi or s > e:
            warnings.append(f"depth{depth} beat '{b.get('label','?')}' range {s}-{e} outside {lo}-{hi}; clamped")
            s, e = max(s, lo), min(max(e, lo), hi)
            if s > e:
                s = e = lo
            b["start"], b["end"] = s, e
        if s != prev_end + 1:
            warnings.append(f"depth{depth} beat '{b.get('label','?')}' starts at {s}, expected {prev_end + 1} (gap/overlap)")
        prev_end = max(prev_end, e)
        seg = range(s - 1, e)
        words = sum(len(lines[i].split()) for i in seg)
        b["word_count"] = words
        attributed = [(len(lines[i].split()), speakers[i]) for i in seg if speakers[i]]
        aw = sum(w for w, _ in attributed)
        named = has_named_labels(sp for _, sp in attributed)
        b["tina_share"] = round(sum(w for w, sp in attributed if is_tina(sp)) / aw, 3) if aw and named else None
        if b.get("children"):
            check_and_enrich_beats(b["children"], lines, speakers, warnings, s, e, depth + 1)
    if beats and prev_end < hi:
        warnings.append(f"depth{depth} beats end at {prev_end}, file range ends {hi} (uncovered tail)")


def main():
    os.makedirs(OUT, exist_ok=True)
    excluded = load_excluded()
    index = load_index()
    manifest = []
    for path in sorted(glob.glob(os.path.join(HERE, "data", "subjects", "*.json"))):
        with open(path) as f:
            ext = json.load(f)
        tid = ext["transcript"]
        if tid in excluded:
            vp = os.path.join(OUT, f"{tid}.json")
            if os.path.exists(vp):
                os.remove(vp)
            continue
        meta = index.get(tid)
        if not meta:
            print(f"!! {tid}: not in index, skipping")
            continue
        tpath = os.path.join(VAULT, "transcripts", meta["path"])
        if not os.path.exists(tpath):
            print(f".. skip {tid}: source file not in share ({meta['path']})")
            continue
        with open(tpath, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        speakers, speaker_mode = attribute_speakers(lines)

        subjects = [s for s in ext.get("subjects", []) if s.get("mentions")]
        kept, dropped = verify_mentions(subjects, lines)
        subjects = [s for s in subjects if s["mentions"]]
        subjects.sort(key=lambda s: -len(s["mentions"]))

        beats, beat_warnings, tina_evidence = None, [], None
        bpath = os.path.join(HERE, "data", "beats", f"{tid}.json")
        if os.path.exists(bpath):
            with open(bpath) as f:
                bext = json.load(f)
            beats = bext.get("beats", [])
            tina_evidence = bext.get("tina_evidence")
            check_and_enrich_beats(beats, lines, speakers, beat_warnings)

        total_words = sum(len(l.split()) for l in lines)
        attributed = [(len(lines[i].split()), speakers[i]) for i in range(len(lines)) if speakers[i]]
        aw = sum(w for w, _ in attributed)
        named = has_named_labels(sp for _, sp in attributed)
        tina_share = round(sum(w for w, sp in attributed if is_tina(sp)) / aw, 3) if aw and named else None

        out = {
            "id": tid,
            "title": meta["title"] or tid,
            "date": meta["date"],
            "meeting_type": meta["meeting_type"],
            "notes": ext.get("notes", ""),
            "lines": lines,
            "speakers": speakers,
            "speaker_mode": speaker_mode,
            "tina_share": tina_share,
            "tina_evidence": tina_evidence,
            "subjects": subjects,
            "beats": beats,
            "beat_warnings": beat_warnings,
            "verify": {"kept": kept, "dropped": dropped},
        }
        with open(os.path.join(OUT, f"{tid}.json"), "w") as f:
            json.dump(out, f)
        manifest.append({
            "id": tid, "title": out["title"], "date": out["date"],
            "meeting_type": out["meeting_type"], "n_subjects": len(subjects),
            "n_mentions": kept, "dropped": dropped,
            "has_beats": beats is not None, "words": total_words,
            "tina_share": tina_share,
        })
        bw = f", {len(beat_warnings)} beat warnings" if beat_warnings else ""
        bs = f", beats={'yes' if beats is not None else 'no'}"
        print(f"{tid}: {len(subjects)} subjects, {kept} mentions ({dropped} dropped){bs}"
              f"{bw}, speakers={speaker_mode}, tina_share={tina_share}")
    manifest.sort(key=lambda m: m["date"])
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest: {len(manifest)} transcripts")


if __name__ == "__main__":
    main()
