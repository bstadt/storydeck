#!/usr/bin/env python3
"""Pairwise 'entanglement' between active subjects over time, for the group-geometry
force graph. Entanglement blends three signals:

  - ROOMS  (transcript co-occurrence): they show up in the same conversations
    (Jaccard of their transcript sets).
  - VOICE  (speaker-weighted beat overlap): they're grounded in the same narrative
    beat AND one of them actually speaks it — a subject bound to whoever discusses it.
  - RELATION (proximity-clustered co-mention): a speaker asserts the two together in
    a focused statement ("Ada is working on the shared OS", "Ben is a heavy tool
    seed user"). We cluster each line's entity mentions by proximity and weight a pair
    by 1/(entities in its cluster − 1) — so a focused 2-entity relation counts fully,
    while a comma-separated shortlist of names barely counts.

Computed per month (windowed) so the graph's forces evolve. Exports
viewer/data/entanglement.json.
"""
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VD = os.path.join(HERE, "viewer", "data")

# blend weights: rooms + voice + relation
W_TX, W_VOICE, W_REL = 0.4, 0.25, 0.35
EDGE_FLOOR = 0.04
PROX = 160      # chars: mentions within this span on a line are one relational cluster
REL_TAU = 1.5   # saturation of accumulated relational co-mention


def month(date):
    return date[:7]


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def beat_score(bA, bB, vA, vB):
    """Fraction of shared beats in which one of the pair actually SPEAKS."""
    inter = bA & bB
    if not inter:
        return 0.0
    s = sum(1.0 for k in inter if (k in vA or k in vB))
    m = min(len(bA), len(bB))
    return s / m if m else 0.0


def saturate(raw):
    return 1.0 - math.exp(-raw / REL_TAU)


def entanglement(txA, txB, bA, bB, vA, vB, rel):
    tj = jaccard(txA, txB)
    bo = beat_score(bA, bB, vA, vB)
    return round(W_TX * tj + W_VOICE * bo + W_REL * rel, 4), len(txA & txB), len(bA & bB)


def build_relation(TX, active, canon):
    """rel_month[(a,b)][month] and rel_all[(a,b)] from proximity-clustered co-mentions."""
    active_set = set(active)
    rel_month = defaultdict(lambda: defaultdict(float))
    rel_all = defaultdict(float)
    for tx in TX:
        m = month(tx["date"])
        vp = os.path.join(VD, f"{tx['id']}.json")
        if not os.path.exists(vp):
            continue
        comp = json.load(open(vp))
        lines = comp.get("lines", [])
        per_line = defaultdict(list)          # line -> [(cid, char_pos)]
        for s in comp.get("subjects", []):
            cid = canon.get(s["id"], s["id"])
            if cid not in active_set:
                continue
            for mn in s.get("mentions", []):
                ln, q = mn.get("line"), (mn.get("quote") or "")
                if not ln or ln - 1 >= len(lines):
                    continue
                lt = (lines[ln - 1] or "").lower()
                pos = lt.find(q.lower().strip()[:32]) if q else -1
                per_line[ln].append((cid, pos if pos >= 0 else 0))
        for ms in per_line.values():
            ms.sort(key=lambda x: x[1])
            cluster = [ms[0]]
            clusters = []
            for prev, cur in zip(ms, ms[1:]):
                (cluster.append(cur) if cur[1] - prev[1] <= PROX
                 else (clusters.append(cluster), cluster := [cur]))
            clusters.append(cluster)
            for cl in clusters:
                cids = {c for c, _ in cl}
                if len(cids) < 2:
                    continue
                wl = 1.0 / (len(cids) - 1)     # 2 entities → 1.0 ; a 5-way list → 0.25
                for a, b in itertools.combinations(sorted(cids), 2):
                    rel_month[(a, b)][m] += wl
                    rel_all[(a, b)] += wl
    return rel_month, rel_all


def mds_init(active, tx_of, beat_of, voice_of, rel_all):
    """Classical MDS on the all-time entanglement → a stable 2D 'home' layout."""
    n = len(active)
    allof = lambda d, c: (set().union(*d[c].values()) if d[c] else set())
    S = np.zeros((n, n))
    for ii in range(n):
        for jj in range(ii + 1, n):
            a, b = active[ii], active[jj]
            key = (a, b) if a < b else (b, a)
            w, _, _ = entanglement(allof(tx_of, a), allof(tx_of, b), allof(beat_of, a),
                                   allof(beat_of, b), voice_of[a], voice_of[b],
                                   saturate(rel_all.get(key, 0.0)))
            S[ii, jj] = S[jj, ii] = w
    smax = S.max() or 1.0
    D = np.nan_to_num(1.0 - S / smax, nan=1.0, posinf=1.0, neginf=1.0)
    np.fill_diagonal(D, 0.0)
    J = np.eye(n) - np.ones((n, n)) / n
    with np.errstate(all="ignore"):          # numpy 2.0 emits spurious FP warnings in matmul here
        B = -0.5 * J @ (D ** 2) @ J
        vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:2]
    coords = vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))
    coords -= coords.mean(0)
    coords /= (np.abs(coords).max() or 1.0)
    # deterministic tiny jitter (golden-angle) so no two subjects start coincident,
    # which would leave them stuck (repulsion direction is undefined at distance 0)
    ga = 2.399963229
    for i in range(n):
        coords[i, 0] += 0.04 * math.cos(i * ga)
        coords[i, 1] += 0.04 * math.sin(i * ga)
    return coords.tolist()


def centrality_series(n, edges_by_month):
    """Per-node eigenvector centrality (influence) for each month, normalized to
    [0,1] across the whole series. Influence = being entangled with influential
    subjects, not just having many edges."""
    series = [[0.0] * len(edges_by_month) for _ in range(n)]
    for mi, links in enumerate(edges_by_month):
        A = np.zeros((n, n))
        for i, j, w, *_ in links:
            A[i, j] = A[j, i] = w
        if A.sum() == 0:
            continue
        v = np.ones(n) / np.sqrt(n)
        with np.errstate(all="ignore"):
            for _ in range(300):
                nv = A @ v
                nrm = np.linalg.norm(nv)
                if nrm == 0:
                    break
                nv = nv / nrm
                if np.abs(nv - v).sum() < 1e-10:
                    v = nv
                    break
                v = nv
        v = np.abs(v)
        for i in range(n):
            series[i][mi] = float(v[i])
    mx = max((max(r) for r in series), default=1.0) or 1.0
    return [[c / mx for c in r] for r in series]


def main():
    show = json.load(open(os.path.join(VD, "showcase.json")))
    TX, SUB = show["transcripts"], show["subjects"]
    reg = json.load(open(os.path.join(VD, "registry.json")))["subjects"]
    canon = json.load(open(os.path.join(HERE, "data", "canonical_vocab.json")))["canon"]
    active = [c for c in reg if reg[c]["status"] == "active" and c in SUB]
    active.sort(key=lambda c: -SUB[c]["n_tx"])

    tx_of = {c: {} for c in active}     # c -> {month: set(ti)}
    beat_of = {c: {} for c in active}   # c -> {month: set("ti/path")}
    voice_of = {c: set(SUB[c].get("voice", [])) for c in active}
    for c in active:
        for h in SUB[c]["hits"]:
            ti = h[0]; m = month(TX[ti]["date"])
            tx_of[c].setdefault(m, set()).add(ti)
            beat_of[c].setdefault(m, set()).add("/".join(map(str, h)))
    months = sorted({month(TX[ti]["date"]) for c in active for ti in
                     (set().union(*tx_of[c].values()) if tx_of[c] else set())})
    rel_month, rel_all = build_relation(TX, active, canon)

    def rel(a, b, m):
        key = (a, b) if a < b else (b, a)
        return saturate(rel_month.get(key, {}).get(m, 0.0))

    subjects = [{"id": c, "display": SUB[c]["display"], "type": SUB[c].get("type", "concept"),
                 "color": reg[c].get("color", "#888"), "n_tx": SUB[c]["n_tx"]} for c in active]
    idx = {c: i for i, c in enumerate(active)}
    presence = [[len(tx_of[c].get(m, set())) for m in months] for c in active]

    edges_by_month = []
    for m in months:
        links = []
        for a, b in itertools.combinations(active, 2):
            w, sh, sb = entanglement(tx_of[a].get(m, set()), tx_of[b].get(m, set()),
                                     beat_of[a].get(m, set()), beat_of[b].get(m, set()),
                                     voice_of[a], voice_of[b], rel(a, b, m))
            if w >= EDGE_FLOOR:
                links.append([idx[a], idx[b], w, sh, sb])
        edges_by_month.append(links)

    out = {"subjects": subjects, "months": months, "presence": presence,
           "edges": edges_by_month,
           "centrality": centrality_series(len(active), edges_by_month),
           "init": mds_init(active, tx_of, beat_of, voice_of, rel_all)}
    json.dump(out, open(os.path.join(VD, "entanglement.json"), "w"))

    # --- diagnostics ---
    print(f"{len(active)} active subjects · {len(months)} months")
    allpairs = []
    for a, b in itertools.combinations(active, 2):
        txA = set().union(*tx_of[a].values()) if tx_of[a] else set()
        txB = set().union(*tx_of[b].values()) if tx_of[b] else set()
        beA = set().union(*beat_of[a].values()) if beat_of[a] else set()
        beB = set().union(*beat_of[b].values()) if beat_of[b] else set()
        key = (a, b) if a < b else (b, a)
        w, sh, sb = entanglement(txA, txB, beA, beB, voice_of[a], voice_of[b],
                                 saturate(rel_all.get(key, 0.0)))
        if w >= EDGE_FLOOR:
            allpairs.append((w, round(saturate(rel_all.get(key, 0.0)), 2), a, b))
    allpairs.sort(reverse=True)
    print("\nstrongest bindings (blended) — [rel] = relational co-mention component:")
    for w, r, a, b in allpairs[:16]:
        print(f"  {w:.3f}  [rel {r:.2f}]  {SUB[a]['display']}  ×  {SUB[b]['display']}")
    print("\nedges per month:", {m: len(e) for m, e in zip(months, edges_by_month)})


if __name__ == "__main__":
    main()
