#!/usr/bin/env python3
"""Instance configuration: everything corpus-specific in one file.

Storydeck code is generic; `instance.json` binds it to a particular vault of
transcripts (program framing for prompts, the subject roster with ASR alias
stems, canonical-id aliases, excluded pseudo-subjects). Point the
STORYDECK_INSTANCE env var at another file to run on a different corpus.

Prompt templates in gen_*/serve use {PROGRAM}, {PROGRAM_SHORT}, {SPAN_FULL},
{SPAN_ACTIVE}, {SEASON} placeholders; fill_program() substitutes them at
module import (plain .replace — no brace-escaping issues with .format fields).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_cfg = None


def cfg():
    global _cfg
    if _cfg is None:
        path = os.environ.get("STORYDECK_INSTANCE", os.path.join(HERE, "instance.json"))
        _cfg = json.load(open(path))
    return _cfg


def subjects():
    """{canonical-id: {"stems": [...], "exclude": [...]}} — the tracked roster."""
    return cfg()["subjects"]


def subject_aliases():
    """Legacy/alternate canonical ids -> canonical id."""
    return cfg().get("subject_aliases", {})


def excluded_subjects():
    """Pseudo-subjects to drop from graphs/partitions (e.g. 'redaction')."""
    return set(cfg().get("exclude_subjects", []))


def program():
    return cfg()["program"]


def fill_program(template):
    """Substitute {PROGRAM}/{PROGRAM_SHORT}/{SPAN_FULL}/{SPAN_ACTIVE}/{SEASON}."""
    p = program()
    return (template
            .replace("{PROGRAM_SHORT}", p["short_name"])
            .replace("{PROGRAM}", p["name"])
            .replace("{SPAN_FULL}", p["span_full"])
            .replace("{SPAN_ACTIVE}", p["span_active"])
            .replace("{SEASON}", p["season"]))
