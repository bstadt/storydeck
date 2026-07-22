#!/usr/bin/env python3
"""Weekly cohort grouping BY THE MODEL: one call per meeting-week, in
chronological order. Each call sees the FULL meeting history up to and through
that week (summaries + triples, as an incrementally-growing prompt-cached
system block — no future leakage) and must EXHAUSTIVELY assign every roster
subject to exactly one working group or explicitly to "inactive".

This replaces algorithmic community detection for the geometry tab's weekly
communities — the model reads who actually met/worked together that week.

Output: data/facts/week_groups.json =
  { "<monday-iso>": {"groups": [{"members": [keys], "label": str, "desc": str}],
                     "inactive": [keys]} }
"""
import datetime as dt
import glob
import json
import os
import re
import time
import instance_config as _ic

import cohort_axis as ax
import extract_facts as ef
import gen_story_triplets as g

MODEL = "claude-opus-4-8"
EXCLUDE = _ic.excluded_subjects()   # pseudo-subjects dropped from the graph

# week-agnostic preamble: the system block must be a byte-stable, append-only
# prefix across calls so every week's call extends the same prompt cache
SYSTEM_HEAD = _ic.fill_program("""You are mapping WHO IS ACTUALLY WORKING WITH WHOM in the {PROGRAM}. Below is the meeting history in chronological order — each meeting's summary and mined (subject -> relation -> object) triples. You will be asked to partition the cohort for the FINAL week shown; earlier meetings are context for understanding ongoing collaborations and disambiguating names.

## MEETING HISTORY (chronological)

""")

TASK = """The current week is {week_label} (week of {monday}). This week's meetings are the ones dated {monday} through {sunday} at the END of the history above.

## Cohort subjects (canonical keys with name variants)
{roster}

## Task
Assign EVERY subject in the roster above to EXACTLY ONE of:
- a WORKING GROUP: subjects genuinely working together THIS WEEK (same meetings, joint work, direct hand-offs). A subject actively working alone this week gets its own single-member group. Base membership on THIS WEEK's activity only — history is context, not evidence of current activity.
- "inactive": not present/participating in any of this week's meetings. Being merely mentioned, discussed, or evaluated in absence counts as inactive.

Every roster key must appear exactly once across groups + inactive. Use ONLY canonical keys.

Output ONLY JSON:
{{"groups": [{{"members": ["<key>", ...], "label": "<2-5 word group name>", "desc": "<one sentence: what this group worked on together this week>"}}], "inactive": ["<key>", ...]}}"""


def roster():
    lines = []
    for k, spec in g.SUBJECTS.items():
        if k in EXCLUDE:
            continue
        lines.append(f"- {k}  (aka: {', '.join(spec['stems'])})")
    return "\n".join(lines)


def week_blocks():
    """monday-iso -> [(date, block string)], blocks date-sorted within the week"""
    by_week = {}
    for jf in glob.glob("data/facts/*.json"):
        tid = os.path.basename(jf)[:-5]
        if tid.endswith((".orig", ".insights")):
            continue
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        if not isinstance(d, dict) or "triples" not in d:
            continue
        m = ef.lookup_meta(tid)
        ds = (m["date"] or "")[:10]
        try:
            day = dt.date.fromisoformat(ds)
        except ValueError:
            continue
        monday = (day - dt.timedelta(days=day.weekday())).isoformat()
        rows = "\n".join(f"{t['subject']} -> {t['relation']} -> {t['object']}"
                         for t in d["triples"])
        by_week.setdefault(monday, []).append(
            (ds, tid, f"### [{ds}] {m['title']} ({m['mtype']})\n{d.get('summary','')}\n{rows}"))
    return {wk: [b for _, _, b in sorted(v)] for wk, v in by_week.items()}


def parse_partition(txt, keys):
    """-> (groups, inactive, missing). Dedup: first assignment of a key wins."""
    try:
        data = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    except Exception:
        return [], [], set(keys)
    taken, groups = set(), []
    for gr in data.get("groups", []):
        mem = [x for x in (gr.get("members") or []) if x in keys and x not in taken]
        taken.update(mem)
        if mem:
            groups.append({"members": sorted(mem),
                           "label": gr.get("label", ""),
                           "desc": gr.get("desc", "")})
    inactive = sorted(x for x in set(data.get("inactive") or []) if x in keys and x not in taken)
    missing = set(keys) - taken - set(inactive)
    return groups, inactive, missing


def main():
    ef.load_env()
    import anthropic
    client = anthropic.Anthropic(max_retries=3, timeout=600)
    blocks = week_blocks()
    ros = roster()
    weeks = ax.weeks()
    keys = set(g.SUBJECTS) - EXCLUDE

    try:
        out = json.load(open("data/facts/week_groups.json"))
    except Exception:
        out = {}
    # drop entries from the old format (no exhaustive inactive list)
    out = {wk: v for wk, v in out.items() if "inactive" in v}

    def call(messages, history, salt):
        """One streamed call. salt: trailing newlines on the last user turn —
        a byte-identical prompt can hit a poisoned server cache entry that
        500s deterministically; any perturbation routes around it.

        history is a LIST of text parts (head + one part per week) and is sent
        as separate system blocks: cache lookup only matches at block
        boundaries, so a single ever-growing block never hits cache — per-week
        blocks let call k+1 hit the boundary call k cached one block back."""
        msgs = [dict(m) for m in messages]
        msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + "\n" * salt}
        system = [{"type": "text", "text": t} for t in history]
        system[-1]["cache_control"] = {"type": "ephemeral"}
        with client.messages.stream(
                model=MODEL, max_tokens=4000, thinking={"type": "disabled"},
                system=system, messages=msgs) as st:
            msg = st.get_final_message()
        txt = next((b.text for b in msg.content if b.type == "text"), "")
        u = msg.usage
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cost = (u.input_tokens * 5 + cw * 6.25 + cr * 0.5 + u.output_tokens * 25) / 1e6
        return txt, cost

    def safe(messages, history):
        for attempt in range(4):
            try:
                return call(messages, history, salt=attempt + 1)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(20 * (attempt + 1))

    total, hist_parts = 0.0, [SYSTEM_HEAD]
    for wk, wlabel, _cnt in weeks:
        bl = blocks.get(wk, [])
        if bl:   # one system block per week, append-only: cache prefix grows
            hist_parts.append("".join(b + "\n\n" for b in bl))
        if wk in out:
            continue
        if not bl:
            out[wk] = {"groups": [], "inactive": sorted(keys)}
            print(f"  {wk}: no meetings -> all inactive", flush=True)
            continue
        sunday = (dt.date.fromisoformat(wk) + dt.timedelta(days=6)).isoformat()
        history = list(hist_parts)
        task = TASK.format(week_label=wlabel, monday=wk, sunday=sunday, roster=ros)
        msgs = [{"role": "user", "content": task}]
        txt, cost = safe(msgs, history)
        total += cost
        groups, inactive, missing = parse_partition(txt, keys)
        if missing:   # one repair round: same cache, model completes the partition
            msgs += [{"role": "assistant", "content": txt},
                     {"role": "user", "content":
                      "You omitted these subjects: " + ", ".join(sorted(missing))
                      + ". Reply with the COMPLETE corrected JSON — every roster "
                        "key exactly once across groups + inactive."}]
            txt2, cost2 = safe(msgs, history)
            total += cost2
            g2, i2, m2 = parse_partition(txt2, keys)
            if len(m2) < len(missing):
                groups, inactive, missing = g2, i2, m2
        if missing:   # still incomplete: default the stragglers to inactive
            print(f"    ! {wk}: defaulting to inactive: {sorted(missing)}", flush=True)
            inactive = sorted(set(inactive) | missing)
        out[wk] = {"groups": groups, "inactive": inactive}
        json.dump(out, open("data/facts/week_groups.json", "w"), indent=1)
        na = sum(len(gr["members"]) for gr in groups)
        print(f"  {wk}: {len(groups)} groups / {na} active / {len(inactive)} inactive  (${cost:.2f})", flush=True)
    json.dump(out, open("data/facts/week_groups.json", "w"), indent=1)
    print(f"total ${total:.2f} -> data/facts/week_groups.json ({len(out)} weeks)")


if __name__ == "__main__":
    main()
