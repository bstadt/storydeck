#!/usr/bin/env python3
"""Resolve person-subjects to their speaker identity and find the beats where
they actually SPEAK (not just where they're discussed).

A person's story (a coordinator, a founder…) is dominated by what they say across the
program, which the discussed-as-subject tagging misses entirely. Where the
transcript has named speaker labels, we can attribute word-share per beat and
surface a person's high-share beats as extra ("voice") nodes.
"""
import re

VOICE_SHARE = 0.30   # a beat counts as the person's "voice" if they own >=30% of its attributed words
STOP = {"the", "team", "os", "app", "project", "tokens", "labs", "research", "and"}


def name_tokens(display):
    return {t for t in re.split(r"[^a-z0-9]+", display.lower()) if len(t) > 2 and t not in STOP}


def speaker_tokens(name):
    return {t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if len(t) > 2}


def resolve_speakers(display, speaker_names):
    """Speaker labels that plausibly are this person (share a distinctive name token)."""
    toks = name_tokens(display)
    if not toks:
        return set()
    out = set()
    for sp in speaker_names:
        if toks & speaker_tokens(sp):
            out.add(sp)
    return out


def beat_voice_share(lines, speakers, start, end, matched):
    """Fraction of a beat's attributed words spoken by `matched` speaker labels."""
    total = mine = 0
    for i in range(start - 1, min(end, len(lines))):
        sp = speakers[i] if i < len(speakers) else None
        if not sp:
            continue
        w = len(lines[i].split())
        total += w
        if sp in matched:
            mine += w
    return (mine / total) if total else 0.0


def voice_beats(display, lines, speakers, beats_flat):
    """beats_flat: list of (path, start, end). Returns [(path, share)] for beats
    this person speaks in above threshold. Empty if the transcript has no usable
    speaker labels for them."""
    speaker_names = {s for s in speakers if s}
    matched = resolve_speakers(display, speaker_names)
    if not matched:
        return []
    out = []
    for path, s, e in beats_flat:
        sh = beat_voice_share(lines, speakers, s, e, matched)
        if sh >= VOICE_SHARE:
            out.append((path, round(sh, 3)))
    return out
