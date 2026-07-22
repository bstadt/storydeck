#!/usr/bin/env python3
"""Shared time axes for the cohort visualization, derived from the corpus meeting
dates so the estimator and the renderer agree. Weekly axis = Monday-anchored ISO
weeks that actually contain a meeting (no dead air); monthly axis = its rollup."""
import datetime as dt
import glob
import os

import extract_facts as ef

_MON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _meeting_days():
    days = []
    for jf in glob.glob("data/facts/*.json"):
        tid = os.path.basename(jf)[:-5]
        if tid.endswith((".orig", ".insights")):
            continue
        ds = (ef.lookup_meta(tid).get("date") or "")[:10]
        if len(ds) == 10:
            try:
                days.append(dt.date.fromisoformat(ds))
            except ValueError:
                pass
    return days


def weeks():
    """[(key=Monday-iso, label, count)] for weeks with >=1 meeting, chronological."""
    cnt = {}
    for d in _meeting_days():
        mon = d - dt.timedelta(days=d.weekday())
        cnt[mon] = cnt.get(mon, 0) + 1
    out = []
    for mon in sorted(cnt):
        lab = f"{_MON[mon.month]} {mon.day}"
        if mon.year == 2025:
            lab += " '25"
        out.append((mon.isoformat(), lab, cnt[mon]))
    return out


def months():
    """[(key='YYYY-MM', label)] for months with >=1 meeting, chronological."""
    seen = {}
    for d in _meeting_days():
        seen[f"{d.year:04d}-{d.month:02d}"] = d
    out = []
    for k in sorted(seen):
        y, m = k.split("-")
        lab = _MON[int(m)] + (" '25" if y == "2025" else "")
        out.append((k, lab))
    return out


def week_to_month(week_key):
    return week_key[:7]


if __name__ == "__main__":
    print("weeks:", [w[1] for w in weeks()])
    print("months:", [m[1] for m in months()])
