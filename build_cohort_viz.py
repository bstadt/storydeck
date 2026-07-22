#!/usr/bin/env python3
"""Turn the per-subject DIRECTED, ranked relation estimates into a time series of
graphs and render a single self-contained cohort.html (no server, no deps).

Key modeling choices:
- DIRECTED: each subject's read of "who I'm related to" is kept as a directed
  edge (src -> dst). So every node has an out-strength (who it reaches toward)
  and an in-strength (who reaches toward it) — that asymmetry drives centrality.
- INVERSE-RANK WEIGHTING: within a subject's ranked list, the tie at rank r is
  scaled by 1/r, so each node's top relationships dominate and the incidental
  tail recedes (declutters the graph, sharpens structure).
- WEEKLY axis (meeting-weeks) with a MONTHLY rollup; the viewer toggles between.

Renderer encodes: node size = weighted degree (in+out); nodes are a uniform
neon color (balance coloring retired); translucent hulls = time-varying
communities (stable id => stable color); tooltip = in/out/centrality.
"""
import json
import math

import cohort_axis as ax
import instance_config as _ic


def layout(ids, edges, iters=600):
    """Fruchterman-Reingold with cooling (deterministic hash seed). edges: (i,j,w)."""
    N = len(ids)
    pos = [[0.0, 0.0] for _ in range(N)]
    for i, nid in enumerate(ids):
        h = 2166136261
        for ch in nid:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        a = (h % 1000) / 1000 * 2 * math.pi
        rad = 0.25 + ((h >> 10) % 1000) / 1000 * 0.75
        pos[i] = [math.cos(a) * rad, math.sin(a) * rad]
    k = math.sqrt(1.0 / max(1, N))
    for it in range(iters):
        t = 0.10 * (1 - it / iters) + 1e-3
        disp = [[0.0, 0.0] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx, dy = pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]
                d2 = dx * dx + dy * dy + 1e-6
                f = k * k / d2
                disp[i][0] += dx * f
                disp[i][1] += dy * f
        for (i, j, w) in edges:
            dx, dy = pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]
            av = math.sqrt(dx * dx + dy * dy) / k * (0.4 + 0.9 * w)
            disp[i][0] += dx * av
            disp[i][1] += dy * av
            disp[j][0] -= dx * av
            disp[j][1] -= dy * av
        for i in range(N):
            disp[i][0] -= pos[i][0] * 0.02
            disp[i][1] -= pos[i][1] * 0.02
            dl = math.sqrt(disp[i][0] ** 2 + disp[i][1] ** 2) + 1e-9
            step = min(dl, t) / dl
            pos[i][0] += disp[i][0] * step
            pos[i][1] += disp[i][1] * step
    xs, ys = [p[0] for p in pos], [p[1] for p in pos]
    mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)
    return [[(p[0] - mnx) / max(1e-6, mxx - mnx),
             (p[1] - mny) / max(1e-6, mxy - mny)] for p in pos]


def layout_clustered(ids, edges, primary, iters=700):
    """Community-aware Fruchterman-Reingold: (1) lay out cluster CENTROIDS with
    the plain FR over the aggregated inter-cluster graph, (2) place nodes with
    an anchor spring to their cluster centroid, boosted repulsion across
    clusters, and damped attraction on cross-cluster edges — so each primary
    community occupies its own region and hulls minimally overlap."""
    N = len(ids)
    # cluster keys: real communities + one singleton per never-clustered node
    ckey = []
    for i, p in enumerate(primary):
        ckey.append(f"c{p}" if p >= 0 else f"s{i}")
    keys = sorted(set(ckey), key=lambda k: (k[0] != 'c', k))
    kidx = {k: j for j, k in enumerate(keys)}
    cid = [kidx[k] for k in ckey]
    # meta graph: aggregated inter-cluster weights
    agg = {}
    for a, b, w in edges:
        ca, cb = cid[a], cid[b]
        if ca == cb or w <= 0:
            continue
        p = (min(ca, cb), max(ca, cb))
        agg[p] = agg.get(p, 0.0) + w
    mx = max(agg.values()) if agg else 1.0
    meta_edges = [(a, b, w / mx) for (a, b), w in agg.items()]
    cent = layout(keys, meta_edges, iters=500)
    # node init: centroid + deterministic jitter
    pos = []
    for i, nid in enumerate(ids):
        h = 2166136261
        for ch in nid:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        a = (h % 1000) / 1000 * 2 * math.pi
        r = 0.02 + ((h >> 10) % 1000) / 1000 * 0.05
        cx, cy = cent[cid[i]]
        pos.append([cx + math.cos(a) * r, cy + math.sin(a) * r])
    k = math.sqrt(1.0 / max(1, N))
    for it in range(iters):
        t = 0.08 * (1 - it / iters) + 1e-3
        disp = [[0.0, 0.0] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx, dy = pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]
                d2 = dx * dx + dy * dy + 1e-6
                f = k * k / d2 * (1.0 if cid[i] == cid[j] else 2.4)
                disp[i][0] += dx * f
                disp[i][1] += dy * f
        for (a, b, w) in edges:
            dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
            damp = 1.0 if cid[a] == cid[b] else 0.3   # cross-cluster pull damped
            av = math.sqrt(dx * dx + dy * dy) / k * (0.4 + 0.9 * w) * damp
            disp[a][0] += dx * av
            disp[a][1] += dy * av
            disp[b][0] -= dx * av
            disp[b][1] -= dy * av
        for i in range(N):
            disp[i][0] += (cent[cid[i]][0] - pos[i][0]) * 2.2   # anchor spring
            disp[i][1] += (cent[cid[i]][1] - pos[i][1]) * 2.2
            dl = math.sqrt(disp[i][0] ** 2 + disp[i][1] ** 2) + 1e-9
            step = min(dl, t) / dl
            pos[i][0] += disp[i][0] * step
            pos[i][1] += disp[i][1] * step
    xs, ys = [p[0] for p in pos], [p[1] for p in pos]
    mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)
    return [[(p[0] - mnx) / max(1e-6, mxx - mnx),
             (p[1] - mny) / max(1e-6, mxy - mny)] for p in pos]



def separate(pos, rads, pad=0.013, iters=160, aspect=1.4):
    """Size-aware de-crowding post-pass on the finished layout: positions are
    static but nodes render up to ~32px radius at their peak week, so the
    dense core overlaps once sizes grow. Enforce a per-pair spacing floor of
    r_i + r_j + pad (radii normalized to canvas height, aspect-corrected
    space ~ screen px), then renormalize to [0,1]."""
    P = [[p[0] * aspect, p[1]] for p in pos]
    N = len(P)
    for _ in range(iters):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dmin = min(0.16, rads[i] + rads[j] + pad)
                dx, dy = P[j][0] - P[i][0], P[j][1] - P[i][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d >= dmin:
                    continue
                if d < 1e-9:
                    dx, dy, d = 1e-4 * (1 + i), 1e-4 * (1 + j), 1e-4
                push = (dmin - d) / d / 2
                P[i][0] -= dx * push; P[i][1] -= dy * push
                P[j][0] += dx * push; P[j][1] += dy * push
                moved = True
        if not moved:
            break
    xs, ys = [p[0] for p in P], [p[1] for p in P]
    mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)
    return [[(p[0] - mnx) / max(1e-6, mxx - mnx),
             (p[1] - mny) / max(1e-6, mxy - mny)] for p in P]


def communities_multilayer(N, edge_list, atag, btag, nframes, gamma=1.0,
                           omega=0.5, gap_decay=0.7, act_min=0.02, min_conc=2):
    """Multilayer modularity (Mucha et al. 2010) via iterated greedy local moves.

    One supra-graph: a node copy (i, f) exists only where subject i has raw tie
    activity in frame f (the activity gate). Intra-layer edges are that frame's
    symmetric tie weights (with a per-layer Newman-Girvan null model, resolution
    gamma); consecutive copies of the same subject are coupled with omega
    (decayed by gap_decay per skipped frame, no null model). Local moves greedily
    maximize multilayer modularity until convergence (deterministic order).

    Communities are single objects SPANNING frames — stable ids by construction,
    no registry matching, no cap-splitting, no smoothing stack. Granularity is
    controlled by gamma, temporal stickiness by omega. Communities that never
    have >= min_conc concurrent members are dropped (-1).

    Returns (frames, n_ids): frames[f][i] -> community id or -1.
    """
    # per-frame symmetric weights + degrees
    W = [dict() for _ in range(nframes)]
    deg = [[0.0] * N for _ in range(nframes)]
    for e in edge_list:
        for f in range(nframes):
            w = max(e[atag].get(f, 0.0), e[btag].get(f, 0.0))
            if w > 0:
                p = (e["a"], e["b"])
                W[f][p] = W[f].get(p, 0.0) + w
                deg[f][e["a"]] += w
                deg[f][e["b"]] += w
    m2 = [max(1e-9, sum(deg[f])) for f in range(nframes)]   # = 2*m_f
    # node copies (activity-gated)
    copies, ids = {}, []
    for f in range(nframes):
        for i in range(N):
            if deg[f][i] >= act_min:
                copies[(i, f)] = len(ids)
                ids.append((i, f))
    U = len(ids)
    adj = [dict() for _ in range(U)]
    for f in range(nframes):
        for (a, b), w in W[f].items():
            ua, ub = copies.get((a, f)), copies.get((b, f))
            if ua is None or ub is None:
                continue
            adj[ua][ub] = adj[ua].get(ub, 0.0) + w
            adj[ub][ua] = adj[ub].get(ua, 0.0) + w
    for i in range(N):
        fs = [f for f in range(nframes) if (i, f) in copies]
        for fa, fb in zip(fs, fs[1:]):
            w = omega * (gap_decay ** (fb - fa - 1))
            ua, ub = copies[(i, fa)], copies[(i, fb)]
            adj[ua][ub] = adj[ua].get(ub, 0.0) + w
            adj[ub][ua] = adj[ub].get(ua, 0.0) + w
    # greedy local moves on multilayer modularity
    comm = list(range(U))
    cdeg = {}                       # (c, f) -> sum of layer-degrees in community
    for u, (i, f) in enumerate(ids):
        cdeg[(u, f)] = deg[f][i]
    for _sweep in range(80):
        moved = False
        for u in range(U):
            i, f = ids[u]
            cu = comm[u]
            ki = deg[f][i]
            cdeg[(cu, f)] = cdeg.get((cu, f), 0.0) - ki
            links = {}
            for v, w in adj[u].items():
                cv = comm[v]
                links[cv] = links.get(cv, 0.0) + w
            best_c = cu
            best_g = links.get(cu, 0.0) - gamma * ki * cdeg.get((cu, f), 0.0) / m2[f]
            for c in sorted(links):
                if c == cu:
                    continue
                g = links[c] - gamma * ki * cdeg.get((c, f), 0.0) / m2[f]
                if g > best_g + 1e-12:
                    best_c, best_g = c, g
            comm[u] = best_c
            cdeg[(best_c, f)] = cdeg.get((best_c, f), 0.0) + ki
            if best_c != cu:
                moved = True
        if not moved:
            break
    # drop communities that never reach min_conc concurrent members
    conc = {}
    for u, (i, f) in enumerate(ids):
        conc.setdefault(comm[u], {}).setdefault(f, 0)
        conc[comm[u]][f] += 1
    ok = {c for c, per in conc.items() if max(per.values()) >= min_conc}
    # renumber by first appearance (stable, deterministic)
    order, remap = [], {}
    for u in range(U):
        c = comm[u]
        if c in ok and c not in remap:
            remap[c] = len(order)
            order.append(c)
    frames = [[-1] * N for _ in range(nframes)]
    for u, (i, f) in enumerate(ids):
        c = comm[u]
        frames[f][i] = remap.get(c, -1) if c in ok else -1
    return frames, len(order)



def _louvain(N, adj, gamma=1.0):
    """Single-layer weighted Louvain with aggregation (deterministic order).
    adj: {i: {j: w}}. Returns labels[i] (community index, arbitrary ids)."""
    node_map = [[i] for i in range(N)]          # supernode -> original nodes
    labels_of = list(range(N))
    cur_adj = {i: dict(adj.get(i, {})) for i in range(N)}
    while True:
        n = len(cur_adj)
        keys = sorted(cur_adj)
        deg = {i: sum(cur_adj[i].values()) for i in keys}
        m2 = max(1e-12, sum(deg.values()))
        comm = {i: i for i in keys}
        cdeg = {i: deg[i] for i in keys}
        improved = False
        for _ in range(60):
            moved = False
            for i in keys:
                ci = comm[i]
                cdeg[ci] -= deg[i]
                links = {}
                for j, w in cur_adj[i].items():
                    if j == i:
                        continue
                    links[comm[j]] = links.get(comm[j], 0.0) + w
                best_c, best_g = ci, links.get(ci, 0.0) - gamma * deg[i] * cdeg.get(ci, 0.0) / m2
                for c in sorted(links):
                    g = links[c] - gamma * deg[i] * cdeg.get(c, 0.0) / m2
                    if g > best_g + 1e-12:
                        best_c, best_g = c, g
                comm[i] = best_c
                cdeg[best_c] = cdeg.get(best_c, 0.0) + deg[i]
                if best_c != ci:
                    moved = True
                    improved = True
            if not moved:
                break
        # aggregate
        groups = {}
        for i in keys:
            groups.setdefault(comm[i], []).append(i)
        if not improved or len(groups) == n:
            out = [0] * N
            for i in keys:
                for orig in node_map[i]:
                    out[orig] = comm[i]
            return out
        order = sorted(groups)
        gid = {c: k for k, c in enumerate(order)}
        new_map, new_adj = [], {}
        for k, c in enumerate(order):
            members = []
            for i in groups[c]:
                members.extend(node_map[i])
            new_map.append(members)
            new_adj[k] = {}
        for i in keys:
            for j, w in cur_adj[i].items():
                a, b = gid[comm[i]], gid[comm[j]]
                if a == b and i >= j:
                    continue
                new_adj[a][b] = new_adj[a].get(b, 0.0) + w
        node_map, cur_adj = new_map, new_adj


def communities_tracked(N, edge_list, atag, btag, nframes, gamma=0.8,
                        act_min=0.02, min_size=2, match_thresh=0.3, memory=3):
    """Per-frame modularity Louvain + cross-frame community TRACKING.

    Each frame's graph (raw weights, activity-gated) is partitioned by proper
    Louvain at resolution gamma — no giant-blob cap needed, no smoothing stack.
    Communities are then tracked across frames: greedy one-to-one Jaccard
    matching against communities seen in the last `memory` frames (so a group
    survives a quiet week); matched groups keep their stable id, unmatched
    mint new ids. Returns (frames, n_ids)."""
    frames = []
    recent = []                     # list of (frame_idx, {sid: memberset})
    nid = 0
    for f in range(nframes):
        adj = {}
        deg = [0.0] * N
        for e in edge_list:
            w = max(e[atag].get(f, 0.0), e[btag].get(f, 0.0))
            if w > 0:
                adj.setdefault(e["a"], {})[e["b"]] = adj.get(e["a"], {}).get(e["b"], 0.0) + w
                adj.setdefault(e["b"], {})[e["a"]] = adj.get(e["b"], {}).get(e["a"], 0.0) + w
                deg[e["a"]] += w
                deg[e["b"]] += w
        active = [i for i in range(N) if deg[i] >= act_min]
        aset = set(active)
        adj = {i: {j: w for j, w in nb.items() if j in aset}
               for i, nb in adj.items() if i in aset}
        # pure Leiden on this week's graph (proper implementation, deterministic seed)
        import igraph, leidenalg
        vid = {i: k for k, i in enumerate(active)}
        es, ws = [], []
        for i in active:
            for j, w in adj.get(i, {}).items():
                if i < j:
                    es.append((vid[i], vid[j]))
                    ws.append(w)
        groups = {}
        if active:
            g = igraph.Graph(n=len(active), edges=es)
            part = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition,
                weights=ws, resolution_parameter=gamma, seed=42, n_iterations=-1)
            for k, i in enumerate(active):
                groups.setdefault(part.membership[k], set()).add(i)
        comms = sorted((g for g in groups.values() if len(g) >= min_size), key=min)
        # track: greedy best-Jaccard one-to-one vs recently seen communities
        seen = {}
        for _, snap in recent:
            for sid, mem in snap.items():
                seen[sid] = mem            # newest snapshot of each sid wins
        pairs = []
        for ci, cset in enumerate(comms):
            for sid, pset in seen.items():
                inter = len(cset & pset)
                if not inter:
                    continue
                jac = inter / len(cset | pset)
                if jac >= match_thresh:
                    pairs.append((jac, ci, sid))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        used_c, used_s, assign = set(), set(), {}
        for jac, ci, sid in pairs:
            if ci in used_c or sid in used_s:
                continue
            assign[ci] = sid
            used_c.add(ci)
            used_s.add(sid)
        frame = [-1] * N
        snap = {}
        for ci, cset in enumerate(comms):
            sid = assign.get(ci)
            if sid is None:
                sid = nid
                nid += 1
            for i in cset:
                frame[i] = sid
            snap[sid] = set(cset)
        frames.append(frame)
        recent.append((f, snap))
        recent = recent[-memory:]
    return frames, nid



def weekly_groups_from_model(subjects, weeks):
    """Read data/facts/week_groups.json (model-authored weekly partitions) and
    align group ids across weeks (greedy Jaccard vs the last 3 weeks) so a
    persisting group keeps one color. Returns (frames, n_ids, weekGroups) where
    weekGroups[f] = [{id, label, desc, members(names)}]."""
    wg = json.load(open("data/facts/week_groups.json"))
    idx = {s: i for i, s in enumerate(subjects)}
    frames, week_groups, recent, nid = [], [], [], 0
    for wk, _lab, _cnt in weeks:
        entry = wg.get(wk, {"groups": []})
        comms = []
        for gr in entry["groups"]:
            mem = {idx[m] for m in gr["members"] if m in idx}
            if len(mem) >= 1:   # singleton = active-but-solo (model calls the rest inactive)
                comms.append((mem, gr.get("label", ""), gr.get("desc", "")))
        seen = {}
        for snap in recent:
            for sid, mem in snap.items():
                seen[sid] = mem
        pairs = []
        for ci, (cset, _l, _d) in enumerate(comms):
            for sid, pset in seen.items():
                inter = len(cset & pset)
                if inter and inter / len(cset | pset) >= 0.3:
                    pairs.append((inter / len(cset | pset), ci, sid))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        used_c, used_s, assign = set(), set(), {}
        for jac, ci, sid in pairs:
            if ci in used_c or sid in used_s:
                continue
            assign[ci] = sid
            used_c.add(ci)
            used_s.add(sid)
        frame = [-1] * len(subjects)
        snap, out = {}, []
        for ci, (cset, lab, desc) in enumerate(comms):
            sid = assign.get(ci)
            if sid is None:
                sid = nid
                nid += 1
            for i in cset:
                frame[i] = sid
            snap[sid] = set(cset)
            out.append({"id": sid, "label": lab, "desc": desc,
                        "members": sorted(subjects[i] for i in cset)})
        frames.append(frame)
        week_groups.append(sorted(out, key=lambda g2: -len(g2["members"])))
        recent.append(snap)
        recent = recent[-3:]
    return frames, nid, week_groups


def _lpa(adj, seeds, iters=50):
    """Weighted async label propagation, deterministic (fixed node order,
    prefer-current then smallest-label tie-break). adj: list of {j: w}."""
    labels = list(seeds)
    N = len(adj)
    for _ in range(iters):
        changed = False
        for i in range(N):
            nb = adj[i]
            if not nb:
                continue
            score = {}
            for j, w in nb.items():
                lab = labels[j]
                score[lab] = score.get(lab, 0.0) + w
            best = max(score.values())
            cand = sorted(l for l, sw in score.items() if sw >= best - 1e-12)
            new = labels[i] if labels[i] in cand else cand[0]
            if new != labels[i]:
                labels[i] = new
                changed = True
        if not changed:
            break
    return labels


def _split_capped(cset, adj, cap, min_size):
    """Split an oversized community into strongest-tie subgroups of <= cap
    members: capacity-constrained Kruskal — add internal edges strongest-first,
    refusing any union that would grow a component past cap. Deterministic."""
    nodes = sorted(cset)
    parent = {i: i for i in nodes}
    size = {i: 1 for i in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ed = sorted(((w, i, j) for i in nodes for j, w in adj[i].items()
                 if j in cset and i < j),
                key=lambda e: (-e[0], e[1], e[2]))
    for w, i, j in ed:
        ri, rj = find(i), find(j)
        if ri != rj and size[ri] + size[rj] <= cap:
            parent[rj] = ri
            size[ri] += size[rj]
    comps = {}
    for i in nodes:
        comps.setdefault(find(i), set()).add(i)
    return [c for c in comps.values() if len(c) >= min_size]


def communities_over_time(N, edge_list, atag, btag, nframes, start_id=0,
                          min_size=2, jac_thresh=0.25, wmin=0.08,
                          lam=0.55, cap=11, sticky=1.35, act_min=0.02):
    """Per-frame community detection with temporal stability.

    Frame graph = symmetric (weight = max of the two directions at that
    frame), EXPONENTIALLY SMOOTHED across time: adj_t = w_t + lam*adj_{t-1},
    so a community doesn't dissolve just because of one quiet frame. Only
    smoothed ties above wmin bind a community (granularity lever). Each
    frame's label propagation is SEEDED from the previous frame's stable
    assignment (so communities persist rather than rescrambling); communities
    larger than cap are split into strongest-tie subgroups (capacity-capped
    Kruskal); communities smaller than min_size are dropped (-1); ids are
    matched across frames by Jaccard overlap against a registry of every
    community's last-seen membership (so a community that skips a frame
    regains its id/color).

    Returns (frames, next_id): frames[f][node] -> stable id or -1.
    """
    frames = []
    registry = {}                      # stable id -> last-seen member set
    prev_stable = [-1] * N
    prev_raw = [0.0] * N
    smooth = {}                        # (a,b) a<b -> smoothed weight
    nid = start_id
    for f in range(nframes):
        sm = {}
        for p, v in smooth.items():    # carry-over decays; keep sparse
            dv = v * lam
            if dv > 0.004:
                sm[p] = dv
        for e in edge_list:
            w = max(e[atag].get(f, 0.0), e[btag].get(f, 0.0))
            if w > 0:
                p = (e["a"], e["b"])
                sm[p] = sm.get(p, 0.0) + w
        smooth = sm
        # raw (unsmoothed) activity this frame — membership requires CURRENT
        # activity (this week or last); smoothing only stabilizes cluster shape
        rawdeg = [0.0] * N
        for e in edge_list:
            w = max(e[atag].get(f, 0.0), e[btag].get(f, 0.0))
            if w > 0:
                rawdeg[e["a"]] += w
                rawdeg[e["b"]] += w
        active = [rawdeg[i] >= act_min or prev_raw[i] >= act_min for i in range(N)]
        adj = [dict() for _ in range(N)]
        for (a, b), v in sm.items():
            # sticky bonus: pairs that shared a community last frame bind tighter,
            # damping one-week membership flip-flops between adjacent groups
            if prev_stable[a] >= 0 and prev_stable[a] == prev_stable[b]:
                v = v * sticky
            if v > wmin:
                adj[a][b] = v
                adj[b][a] = v
        # seed labels: previous stable id (offset past node-index label space)
        seeds = [N + prev_stable[i] if prev_stable[i] >= 0 else i
                 for i in range(N)]
        labels = _lpa(adj, seeds)
        groups = {}
        for i in range(N):
            if adj[i]:                 # isolated nodes stay unassigned
                groups.setdefault(labels[i], []).append(i)
        comms = []
        for g in groups.values():
            g = [i for i in g if active[i]]   # no ghost membership from smoothing decay
            if len(g) < min_size:
                continue
            if len(g) > cap:
                comms.extend(_split_capped(set(g), adj, cap, min_size))
            else:
                comms.append(set(g))
        comms.sort(key=min)
        # match current communities to known stable ids by best Jaccard
        pairs = []
        for ci, cset in enumerate(comms):
            for sid, pset in registry.items():
                inter = len(cset & pset)
                # majority continuity: an id carries over only if most of the
                # CURRENT membership was already in it (stops a small old id
                # being hijacked by a mostly-new larger cluster)
                if inter and inter / len(cset) >= 0.5:
                    pairs.append((inter / len(cset | pset), ci, sid))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        cur_sid = [-1] * len(comms)
        used_cur, used_sid = set(), set()
        for jc, ci, sid in pairs:
            if jc < jac_thresh or ci in used_cur or sid in used_sid:
                continue
            cur_sid[ci] = sid
            used_cur.add(ci)
            used_sid.add(sid)
        assigned = [-1] * N
        for ci, cset in enumerate(comms):
            if cur_sid[ci] < 0:
                cur_sid[ci] = nid
                nid += 1
            registry[cur_sid[ci]] = cset
            for i in cset:
                assigned[i] = cur_sid[ci]
        prev_stable = assigned
        prev_raw = rawdeg
        frames.append(assigned)
    return frames, nid


EXCLUDE = _ic.excluded_subjects()   # pseudo-subjects dropped from the graph


def main():
    rel = json.load(open("data/facts/cohort_relations.json"))
    subjects = sorted(k for k in rel if k not in EXCLUDE)
    idx = {s: i for i, s in enumerate(subjects)}

    weeks = ax.weeks()
    wkeys = [w[0] for w in weeks]
    wpos = {k: i for i, k in enumerate(wkeys)}
    months = ax.months()
    mkeys = [m[0] for m in months]
    mpos = {k: i for i, k in enumerate(mkeys)}

    # directed[(src_i, dst_i)] = {"w":{wi:rankw}, "m":{mi:rankw}, "rank":r, "why":s}
    directed = {}
    for src, data in rel.items():
        if src in EXCLUDE:
            continue
        si = idx[src]
        for rank, r in enumerate(data.get("relations", []), start=1):
            dst = r.get("entity")
            if dst not in idx or dst == src:
                continue
            # phantom filter: a mere mention (weak tie, single active week) is
            # not a relationship — e.g. one subject characterizing another
            wk_map = r.get("weeks") or {}
            if float(r.get("overall") or 0) <= 0.3 and len(wk_map) <= 1:
                continue
            di = idx[dst]
            rw = 1.0 / rank                     # inverse-rank weight
            wk, mo = {}, {}
            for k, v in (r.get("weeks") or {}).items():
                if v is not None and isinstance(v, (int, float)) and v <= 0.15:
                    continue   # estimator caps mention-only weeks at 0.1 — not interaction
                if k in wpos and isinstance(v, (int, float)):
                    wi = wpos[k]
                    wk[wi] = max(wk.get(wi, 0.0), float(v) * rw)
                    mi = mpos.get(ax.week_to_month(k))
                    if mi is not None:
                        mo[mi] = max(mo.get(mi, 0.0), float(v) * rw)
            if wk:
                directed[(si, di)] = {"w": wk, "m": mo, "rank": rank,
                                      "why": r.get("why", "")}

    # global normalizers (per axis) so a single directed edge value maxes near 1
    normW = max((max(d["w"].values()) for d in directed.values()), default=1.0)
    normM = max((max(d["m"].values()) for d in directed.values() if d["m"]), default=1.0)

    # fold directed reads into unordered-pair edges keeping both directions
    edges = {}
    for (si, di), d in directed.items():
        a, b = (si, di) if si < di else (di, si)
        e = edges.setdefault((a, b), {"aw": {}, "bw": {}, "am": {}, "bm": {},
                                      "dirs": set(), "why": ""})
        e["dirs"].add(si)
        if len(d["why"]) > len(e["why"]):
            e["why"] = d["why"]
        wtag = "aw" if si == a else "bw"
        mtag = "am" if si == a else "bm"
        for wi, v in d["w"].items():
            e[wtag][wi] = round(v / normW, 4)
        for mi, v in d["m"].items():
            e[mtag][mi] = round(v / normM, 4)

    edge_list = []
    for (a, b), e in edges.items():
        edge_list.append({"a": a, "b": b, "aw": e["aw"], "bw": e["bw"],
                          "am": e["am"], "bm": e["bm"],
                          "mutual": len(e["dirs"]) > 1, "why": e["why"]})

    # time-varying communities FIRST (stable ids across frames, per axis) —
    # the layout below is community-aware so hulls minimally overlap
    import os as _os
    if _os.path.exists("data/facts/week_groups.json"):
        commW, nW, weekGroups = weekly_groups_from_model(subjects, weeks)
    else:
        commW, nW = communities_tracked(len(subjects), edge_list, "aw", "bw",
                                        len(weeks))
        weekGroups = None
    commM, nM = communities_tracked(len(subjects), edge_list, "am", "bm",
                                    len(months))

    # each node's PRIMARY community = modal weekly assignment over its lifetime
    primary = []
    for i in range(len(subjects)):
        cnt = {}
        for frame in commW:
            if frame[i] >= 0:
                cnt[frame[i]] = cnt.get(frame[i], 0) + 1
        primary.append(max(cnt, key=lambda c: (cnt[c], -c)) if cnt else -1)

    # community-aware layout: cluster centroids via meta-FR, nodes anchored to
    # their primary community's centroid with extra cross-cluster repulsion
    lay_edges = []
    for e in edge_list:
        allv = list(e["aw"].values()) + list(e["bw"].values())
        lay_edges.append((e["a"], e["b"], max(allv) if allv else 0.0))
    pos = layout_clustered(subjects, lay_edges, primary)
    # de-crowd the core: spacing floor per pair = peak rendered radii + pad
    peak = [0.0] * len(subjects)
    for e in edge_list:
        av = max(e["aw"].values()) if e["aw"] else 0.0
        bv = max(e["bw"].values()) if e["bw"] else 0.0
        peak[e["a"]] += av + bv
        peak[e["b"]] += av + bv
    mdeg = max(peak) or 1.0
    rads = [(5 + 27 * math.sqrt(min(1.0, d / mdeg))) / 620.0 for d in peak]
    pos = separate(pos, rads)
    nodes = [{"id": s, "label": s, "x": round(pos[i][0], 4), "y": round(pos[i][1], 4)}
             for i, s in enumerate(subjects)]

    # max weighted degree (cumulative-final) per axis, for stable node sizing
    def maxdeg(atag, btag):
        deg = [0.0] * len(subjects)
        for e in edge_list:
            av = max(e[atag].values()) if e[atag] else 0.0
            bv = max(e[btag].values()) if e[btag] else 0.0
            deg[e["a"]] += av + bv
            deg[e["b"]] += av + bv
        return max(deg) or 1.0

    # export weekly community membership (+ member-pair "why" clauses) for the
    # LLM labeler (gen_community_labels.py), and embed its labels if present
    comm_export = {}
    for k, frame in enumerate(commW):
        for ni, cid in enumerate(frame):
            if cid < 0:
                continue
            e = comm_export.setdefault(str(cid), {"frames": {}, "counts": {}})
            e["frames"].setdefault(weeks[k][1], []).append(subjects[ni])
            e["counts"][subjects[ni]] = e["counts"].get(subjects[ni], 0) + 1
    for cid, info in comm_export.items():
        cnt = info["counts"]
        info["members"] = sorted(cnt, key=lambda m: -cnt[m])   # most-frequent first
        # whys: prefer ties between FREQUENT members, diversified (<=3 per node),
        # so one alphabetically-early member can't dominate the labeler's evidence
        cands = []
        for e in edge_list:
            if not e.get("why"):
                continue
            a, b = subjects[e["a"]], subjects[e["b"]]
            if a in cnt and b in cnt:
                cands.append((min(cnt[a], cnt[b]), a, b, e["why"]))
        cands.sort(key=lambda x: -x[0])
        used, whys = {}, []
        for score, a, b, why in cands:
            if used.get(a, 0) >= 3 or used.get(b, 0) >= 3:
                continue
            whys.append(f"{a}~{b}: {why}")
            used[a] = used.get(a, 0) + 1
            used[b] = used.get(b, 0) + 1
            if len(whys) >= 12:
                break
        info["whys"] = whys
    json.dump(comm_export, open("data/facts/communities_computed.json", "w"), indent=1)
    try:
        comm_labels = json.load(open("data/facts/community_labels.json"))
    except Exception:
        comm_labels = {}

    payload = {
        "weeks": {"labels": [w[1] for w in weeks]},
        "months": {"labels": [m[1] for m in months]},
        "nodes": nodes, "edges": edge_list,
        "maxDegW": round(maxdeg("aw", "bw"), 4),
        "maxDegM": round(maxdeg("am", "bm"), 4),
        "commW": commW, "commM": commM,
        "commCount": max(nW, nM),
        "commLabels": comm_labels,
        "weekGroups": weekGroups,
    }
    html = HTML.replace("__DATA__", json.dumps(payload))
    open("cohort.html", "w").write(html)
    open("viewer/geometry.html", "w").write(html)   # the Storydeck geometry tab
    print(f"wrote cohort.html + viewer/geometry.html: {len(nodes)} nodes, "
          f"{len(edge_list)} edges, {len(weeks)} weeks / {len(months)} months, "
          f"{nW} week-communities / {nM} month-communities")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Storydeck | Geometry</title>
<style>
  :root{ --bg:#08090c; --panel:rgba(16,18,24,.92); --ink:#e8e6df; --mut:#9a9890;
         --accent:#4d94e8; --amber:#f08a56; --line:#23262e; --up:#2fd39a; --dn:#ef6a68; }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
    font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;overflow:hidden}
  #brand{display:flex;align-items:center;gap:12px;height:30px}
  #brandword{font:700 15px/30px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;letter-spacing:.01em;white-space:nowrap}
  #tabs{display:flex;gap:6px}
  #tabs a,#tabs span{font-size:11.5px;padding:4px 12px;border:1px solid var(--line);
    border-radius:7px;color:var(--mut);text-decoration:none;background:var(--panel);white-space:nowrap}
  #tabs span.cur{background:var(--ink);color:#0b0b0b;border-color:var(--ink);font-weight:600}
  #wrap{position:fixed;inset:0;display:flex;flex-direction:column}
  header{padding:14px 20px 10px;border-bottom:1px solid var(--line);
    display:flex;justify-content:space-between;align-items:center;gap:20px}
  .legend{font-size:11px;color:var(--mut);text-align:right;white-space:nowrap;line-height:1.7}
  .sw{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px;vertical-align:middle}
  #stage{flex:1;position:relative;min-height:0}
  canvas#c{position:absolute;inset:0;width:100%;height:100%}
  #controls{display:flex;align-items:center;gap:16px;padding:12px 20px;
    border-top:1px solid var(--line);background:var(--panel)}
  button{background:#14171e;color:var(--ink);border:1px solid var(--line);
    border-radius:7px;padding:7px 14px;font-size:13px;cursor:pointer}
  button:hover{background:#1c212b}
  #frame{font-variant-numeric:tabular-nums;font-weight:600;min-width:74px;font-size:15px}
  #scrub{flex:1;accent-color:var(--accent);height:4px}
  .seg{display:flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
  .seg button{border:0;border-radius:0;padding:7px 12px}
  .seg button+button{border-left:1px solid var(--line)}
  .seg button.on{background:#1d2634;color:var(--accent)}
  #tip{position:absolute;pointer-events:none;background:#0f1218f2;
    border:1px solid var(--line);border-radius:8px;padding:9px 11px;max-width:290px;
    font-size:12px;opacity:0;transition:opacity .1s;z-index:6}
  #tip b{color:var(--accent)}
  #tip .meta{color:var(--mut);margin:3px 0 5px;font-size:11px}
  #tip .r{color:var(--mut);display:flex;justify-content:space-between;gap:14px}
  #tip .r span:last-child{color:var(--ink);font-variant-numeric:tabular-nums}
  /* right column: influence-over-time line chart */
  #stocks{position:absolute;top:12px;right:14px;bottom:12px;width:min(42%,560px);
    display:flex;flex-direction:column;background:var(--panel);border:1px solid var(--line);
    border-radius:10px;z-index:5;overflow:hidden}
  #charthead{padding:10px 14px 6px;font-size:11.5px;color:var(--mut);
    border-bottom:1px solid var(--line);display:flex;justify-content:space-between}
  #charthead b{color:var(--ink);font-weight:600}
  #chart{flex:1;width:100%;cursor:crosshair}
  .comm{margin-bottom:14px;padding:6px 8px 12px;border-bottom:1px solid rgba(35,38,46,.6);
    border-radius:8px;cursor:default;transition:background .12s}
  .comm.hot{background:rgba(77,148,232,.08)}
  .comm .hd{display:flex;align-items:center;gap:8px;margin-bottom:3px}
  .comm .csw{width:10px;height:10px;border-radius:3px;flex:none}
  .comm .lbl{font-weight:600;font-size:12.5px}
  .comm .mem{color:var(--mut);font-size:11px;margin-bottom:4px;line-height:1.5}
  .comm .desc{font-size:11.5px;line-height:1.55}
</style></head>
<body><div id="wrap">
  <header>
    <div id="brand"><span id="brandword">Storydeck</span>
      <nav id="tabs"><a href="showcase.html">stories</a><span class="cur">geometry</span><a href="query.html">query</a><a href="api/stories.zip" download>export</a></nav>
    </div>
  </header>
  <div id="stage">
    <canvas id="c"></canvas><div id="tip"></div>
    <div id="stocks">
      <div id="charthead"><b>influence over time</b><span>hover a line or a node · drag to scrub</span></div>
      <canvas id="chart"></canvas>
    </div>
  </div>
  <div id="controls">
    <button id="play">▶ Play</button>
    <span id="frame">—</span>
    <input id="scrub" type="range" min="0" max="1" step="0.0005" value="0">
  </div>
</div>
<script>
const D = __DATA__;
const cv=document.getElementById('c'), ctx=cv.getContext('2d'), tip=document.getElementById('tip');
let gran='week', mode='frame', metric='inf', view='edge', t=0, playing=false, hover=-1, hoverComm=-1, W=0, H=0, DPR=1;

const AX={week:D.weeks.labels, month:D.months.labels};
const TAG={week:['aw','bw'], month:['am','bm']};
function NA(g){ return AX[g||gran].length; }
function maxDeg(){ return gran==='week'?D.maxDegW:D.maxDegM; }

function resize(){ DPR=window.devicePixelRatio||1; const r=cv.getBoundingClientRect();
  W=r.width;H=r.height; cv.width=W*DPR;cv.height=H*DPR; ctx.setTransform(DPR,0,0,DPR,0,0); }
window.addEventListener('resize',resize);

let vs=1,vx=0,vy=0,vdrag=null;   // viewport: screen = base*vs + (vx,vy)
function stocksW(){ return Math.min(W*0.42,560)+28; }
function px(){ const mx=70,my=44,RGT=W-stocksW()-40;
  return D.nodes.map(n=>({x:(mx+n.x*(RGT-mx))*vs+vx,y:(my+n.y*(H-my-56))*vs+vy,label:n.label})); }

function valAt(map,tt,n){
  if(mode==='cum'){ let w=0; const hi=Math.min(n-1,Math.floor(tt));
    for(let k=0;k<=hi;k++){const v=map[k]||0; if(v>w)w=v;}
    if(hi<n-1){ let w2=w; const v=map[hi+1]||0; if(v>w2)w2=v; w+=(w2-w)*(tt-hi); }
    return w; }
  const lo=Math.floor(tt),f=tt-lo; const a=map[lo]||0,b=map[Math.min(n-1,lo+1)]||0;
  return a+(b-a)*f; }

// per-frame directed strengths on axis g -> in/out per node + PageRank centrality
function computeG(g,tt){
  const N=D.nodes.length, n=NA(g), [at,bt]=TAG[g];
  const inS=new Array(N).fill(0), outS=new Array(N).fill(0), E=[];
  for(const e of D.edges){
    const ab=valAt(e[at],tt,n), ba=valAt(e[bt],tt,n);
    if(ab<=0&&ba<=0) continue;
    outS[e.a]+=ab; inS[e.b]+=ab; outS[e.b]+=ba; inS[e.a]+=ba;
    E.push({a:e.a,b:e.b,ab,ba,ud:Math.max(ab,ba),mutual:e.mutual,why:e.why});
  }
  // DIRECTED PageRank: influence = received attention along others' ranked
  // reads. (An undirected variant was tried 2026-07-21 and reverted — it
  // lifted strong-outreach nodes but demoted a known top-3 hub below ground truth.)
  let pr=new Array(N).fill(1/N);
  const out=new Array(N).fill(0);
  for(const e of E){ out[e.a]+=e.ab; out[e.b]+=e.ba; }
  for(let it=0;it<40;it++){
    const np=new Array(N).fill(0.15/N);
    for(const e of E){
      if(out[e.a]>0) np[e.b]+=0.85*pr[e.a]*e.ab/out[e.a];
      if(out[e.b]>0) np[e.a]+=0.85*pr[e.b]*e.ba/out[e.b];
    }
    let dang=0; for(let i=0;i<N;i++) if(out[i]===0) dang+=0.85*pr[i]/N;
    for(let i=0;i<N;i++) np[i]+=dang;
    pr=np;
  }
  // absolute influence: PageRank is a SHARE (sums to 1), so an idle node
  // holds a phantom ~1/N teleport baseline and cross-frame deltas lie
  // (join late -> "negative" change vs your idle-week baseline). Mask
  // inactive nodes to 0 and scale by the frame's total tie mass so values
  // are comparable across frames and "not here yet" is genuinely zero.
  let M=0; for(const e of E) M+=e.ud;
  for(let i=0;i<N;i++) pr[i]=(inS[i]+outS[i])>1e-9 ? pr[i]*M : 0;
  return {inS,outS,E,pr};
}
function compute(){ return computeG(gran,t); }

// centrality series per axis (one PageRank per integer frame), cached per axis|mode
const SER={};
function series(g){
  const key=g+'|'+mode;
  if(SER[key]) return SER[key];
  const n=NA(g), frames=[];
  for(let k=0;k<n;k++) frames.push(computeG(g,k).pr);
  return SER[key]={frames};
}
function tFrac(){ return NA()>1 ? t/(NA()-1) : 0; }
function frameOn(g){ return Math.max(0,Math.min(NA(g)-1,Math.round(tFrac()*(NA(g)-1)))); }

const NODE_COL='#4d94e8';             // uniform node color (matches the stories-tab accent)

// ---------- time-varying community hulls ----------
const HULL_PAD=26, HULL_ALPHA=0.12;   // padding px, base fill alpha
const HULL_PAL=[[57,255,176],[167,130,255],[255,110,199],[255,214,66],
                [82,227,255],[255,122,105],[190,255,80],[130,150,255]];
function hullOf(pts){                  // Andrew's monotone chain
  const p=pts.slice().sort((a,b)=>a.x-b.x||a.y-b.y);
  if(p.length<3) return p;
  const cr=(o,a,b)=>(a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x);
  const lo=[],up=[];
  for(const q of p){ while(lo.length>=2&&cr(lo[lo.length-2],lo[lo.length-1],q)<=0)lo.pop(); lo.push(q); }
  for(let i=p.length-1;i>=0;i--){ const q=p[i];
    while(up.length>=2&&cr(up[up.length-2],up[up.length-1],q)<=0)up.pop(); up.push(q); }
  lo.pop(); up.pop(); return lo.concat(up);
}
function drawHulls(P,act){
  const arr=gran==='week'?D.commW:D.commM;
  if(!arr||!arr.length) return;
  const n=arr.length;
  const lo=Math.max(0,Math.min(n-1,Math.floor(t))), hi=Math.min(n-1,lo+1);
  const f=Math.max(0,Math.min(1,t-lo));
  // cross-fade adjacent frames with the fractional playhead so blobs
  // fade in/out and appear to move rather than popping
  const layers=hi>lo?[[lo,1-f],[hi,f]]:[[lo,1]];
  ctx.save(); ctx.lineJoin='round'; ctx.lineCap='round';
  for(const [k,wgt] of layers){
    if(wgt<=0.02) continue;
    const groups={};
    arr[k].forEach((cid,i)=>{ if(cid>=0)(groups[cid]=groups[cid]||[]).push(i); });
    for(const key of Object.keys(groups).sort((a,b)=>a-b)){
      const cid=+key, mem=groups[key];
      if(mem.length<2) continue;
      let ma=0; for(const i of mem) ma+=act(i); ma/=mem.length;
      let al=HULL_ALPHA*wgt*(0.25+0.75*ma);   // faded members -> paler hull
      if(hoverComm>=0) al*= (cid===hoverComm?2.0:0.22);   // panel/graph hover emphasis
      if(al<=0.004) continue;
      const col=HULL_PAL[cid%HULL_PAL.length];
      const h=hullOf(mem.map(i=>P[i]));
      if(h.length<2) continue;
      ctx.beginPath(); ctx.moveTo(h[0].x,h[0].y);
      for(let q=1;q<h.length;q++) ctx.lineTo(h[q].x,h[q].y);
      ctx.closePath();
      // fat round-join stroke + fill = padded soft blob; the stroke/fill
      // overlap band doubles up into a slightly stronger rim (intended)
      const c=`rgba(${col[0]},${col[1]},${col[2]},`;
      ctx.lineWidth=HULL_PAD*2;
      ctx.strokeStyle=c+al.toFixed(4)+')'; ctx.fillStyle=c+al.toFixed(4)+')';
      ctx.stroke(); ctx.fill();
    }
  }
  ctx.restore();
}

let ST=null;
function draw(){
  ctx.clearRect(0,0,W,H);
  const P=px(); ST=compute(); const {inS,outS,E,pr}=ST;
  const md=maxDeg();
  const hiSet=new Set();
  if(hover>=0){ hiSet.add(hover);
    for(const e of E){ if(e.a===hover)hiSet.add(e.b); if(e.b===hover)hiSet.add(e.a); } }
  // in community view, hovering a community dims non-members
  const nA=NA(), kC=Math.max(0,Math.min(nA-1,Math.round(t)));
  const curComm=(gran==='week'?D.commW:D.commM)[kC]||[];
  const commDim=i=> view==='comm' && hoverComm>=0 && curComm[i]!==hoverComm;
  // activity 0..1 (used only to weight community-hull alpha; nodes stay opaque)
  const act=i=>{ const d=inS[i]+outS[i]; return 0.10+0.90*Math.min(1,Math.pow(d/md,0.45)*1.35); };
  // community hulls, behind everything
  if(view==='comm') drawHulls(P,act);
  // edges (all solid; edge view only)
  if(view==='edge') for(const e of E){
    if(e.ud<=0.008) continue;
    const dim=hover>=0 && !(e.a===hover||e.b===hover);
    ctx.beginPath(); ctx.moveTo(P[e.a].x,P[e.a].y); ctx.lineTo(P[e.b].x,P[e.b].y);
    ctx.lineWidth=0.6+6*e.ud;
    let al=0.05+0.7*e.ud; if(dim)al*=0.12;
    ctx.strokeStyle=e.mutual?`rgba(77,148,232,${al})`:`rgba(154,152,144,${al*0.8})`;
    ctx.stroke();
  }
  // node circles (all first, so no circle ever covers a label) —
  // uniform opaque neon; only transient hover dims non-neighbors
  for(let i=0;i<D.nodes.length;i++){
    const p=P[i]; const deg=inS[i]+outS[i];
    const rr=5+27*Math.sqrt(Math.min(1,deg/md));
    const dim=(hover>=0 && !hiSet.has(i)) || commDim(i);
    ctx.globalAlpha=dim?0.22:1;
    let fill=NODE_COL;
    if(view==='comm'){
      const cid=curComm[i];
      fill = cid>=0 ? 'rgb('+HULL_PAL[cid%HULL_PAL.length].join(',')+')' : '#3a4150';
    }
    ctx.beginPath(); ctx.arc(p.x,p.y,rr,0,7); ctx.fillStyle=fill; ctx.fill();
    ctx.lineWidth=i===hover?2.2:1.4;
    ctx.strokeStyle=i===hover?'#fff':'rgba(255,255,255,0.3)'; ctx.stroke();
    ctx.globalAlpha=1;
  }
  // node labels — a second pass ON TOP of every circle
  ctx.textAlign='center'; ctx.textBaseline='top';
  for(let i=0;i<D.nodes.length;i++){
    const p=P[i]; const deg=inS[i]+outS[i];
    const rr=5+27*Math.sqrt(Math.min(1,deg/md));
    const dim=(hover>=0 && !hiSet.has(i)) || commDim(i);
    ctx.globalAlpha=dim?0.22:1;
    ctx.font=(i===hover?'600 ':'')+'12px -apple-system,sans-serif';
    ctx.fillStyle=i===hover?'#fff':'#e8e6df';
    ctx.fillText(p.label,p.x,p.y+rr+3);
    ctx.globalAlpha=1;
  }
}

// ---------- influence-over-time line chart (right column) ----------
const ch=document.getElementById('chart'), cx2=ch.getContext('2d');
let chDrag=false;
function chartRect(){ const r=ch.getBoundingClientRect();
  return {w:r.width,h:r.height,ml:40,mr:14,mt:12,mb:24}; }
function drawChart(){
  const R=chartRect(); if(R.w<40) return;
  if(ch.width!==Math.round(R.w*DPR)){ ch.width=Math.round(R.w*DPR); ch.height=Math.round(R.h*DPR); }
  cx2.setTransform(DPR,0,0,DPR,0,0); cx2.clearRect(0,0,R.w,R.h);
  const S=series(gran), n=NA(), fr=S.frames;
  const w=R.w-R.ml-R.mr, h=R.h-R.mt-R.mb;
  let mx=1e-9; for(const f of fr) for(const v of f) if(v>mx) mx=v;
  const X=k=>R.ml+w*k/Math.max(1,n-1), Y=v=>R.mt+h*(1-v/mx);
  // y grid (values on the same x10 scale the panel used)
  cx2.font='10px -apple-system,sans-serif'; cx2.textAlign='right'; cx2.textBaseline='middle';
  for(let g=0;g<=3;g++){ const v=mx*g/3, y=Y(v);
    cx2.strokeStyle='rgba(232,230,223,0.05)'; cx2.beginPath();
    cx2.moveTo(R.ml,y); cx2.lineTo(R.w-R.mr,y); cx2.stroke();
    cx2.fillStyle='rgba(154,152,144,0.7)'; cx2.fillText((v*10).toFixed(0),R.ml-6,y); }
  // x ticks: sparse frame labels
  cx2.textAlign='center'; cx2.textBaseline='top';
  const step=Math.max(1,Math.ceil(n/6));
  for(let k=0;k<n;k+=step){ cx2.fillStyle='rgba(154,152,144,0.7)';
    cx2.fillText(AX[gran][k],X(k),R.h-R.mb+6); }
  // all lines dim; hovered line bright and drawn last
  const N=D.nodes.length;
  const line=(i,style,lw)=>{ cx2.strokeStyle=style; cx2.lineWidth=lw; cx2.beginPath();
    for(let k=0;k<n;k++){ const x=X(k),y=Y(fr[k][i]); k?cx2.lineTo(x,y):cx2.moveTo(x,y); }
    cx2.stroke(); };
  for(let i=0;i<N;i++) if(i!==hover) line(i,'rgba(77,148,232,0.20)',1);
  // playhead synced with the network's timeline
  const px_=X(Math.max(0,Math.min(n-1,t)));
  cx2.strokeStyle='rgba(232,230,223,0.25)'; cx2.lineWidth=1;
  cx2.beginPath(); cx2.moveTo(px_,R.mt); cx2.lineTo(px_,R.h-R.mb); cx2.stroke();
  if(hover>=0){
    line(hover,'#9cc4ff',1.8);
    const k=Math.max(0,Math.min(n-1,Math.round(t)));
    const vy=Y(fr[k][hover]);
    cx2.fillStyle='#9cc4ff'; cx2.beginPath(); cx2.arc(px_,vy,3,0,7); cx2.fill();
    cx2.textAlign='left'; cx2.textBaseline='bottom'; cx2.font='11px -apple-system,sans-serif';
    const lbl=D.nodes[hover].label+' · '+(fr[k][hover]*10).toFixed(1);
    const tw=cx2.measureText(lbl).width;
    const lx=Math.min(px_+8,R.w-R.mr-tw), ly=Math.max(R.mt+12,vy-6);
    cx2.fillStyle='rgba(8,9,12,0.75)'; cx2.fillRect(lx-3,ly-13,tw+6,15);
    cx2.fillStyle='#e8e6df'; cx2.fillText(lbl,lx,ly);
  }
}
function chartT(ev){ const r=ch.getBoundingClientRect(), R=chartRect();
  const n=NA(); return Math.max(0,Math.min(n-1,(ev.clientX-r.left-R.ml)/(R.w-R.ml-R.mr)*(n-1))); }
ch.addEventListener('mousedown',ev=>{ chDrag=true; playing=false;
  document.getElementById('play').textContent='▶ Play';
  t=chartT(ev); document.getElementById('scrub').value=t/(NA()-1); });
window.addEventListener('mouseup',()=>{ chDrag=false; });
ch.addEventListener('mousemove',ev=>{
  if(chDrag){ t=chartT(ev); document.getElementById('scrub').value=t/(NA()-1); return; }
  const r=ch.getBoundingClientRect(), R=chartRect();
  const S=series(gran), n=NA(), fr=S.frames;
  let mx=1e-9; for(const f of fr) for(const v of f) if(v>mx) mx=v;
  const kf=(ev.clientX-r.left-R.ml)/(R.w-R.ml-R.mr)*(n-1);
  const k0=Math.max(0,Math.min(n-1,Math.floor(kf))), k1=Math.min(n-1,k0+1), f=kf-k0;
  const my=ev.clientY-r.top, h=R.h-R.mt-R.mb;
  let best=-1,bd=9;
  for(let i=0;i<D.nodes.length;i++){
    const v=fr[k0][i]+(fr[k1][i]-fr[k0][i])*Math.max(0,Math.min(1,f));
    const y=R.mt+h*(1-v/mx);
    const d=Math.abs(y-my); if(d<bd){bd=d;best=i;}
  }
  hover=best;
});
ch.addEventListener('mouseleave',()=>{ if(!chDrag) hover=-1; });

function frameLabel(){ const lo=Math.floor(t),f=t-lo,A=AX[gran];
  return f<0.5?A[lo]:A[Math.min(NA()-1,lo+1)]; }
function tick(){
  if(playing){ t+=(NA()-1)/(17*60);
    if(t>=NA()-1){ t=NA()-1; playing=false; document.getElementById('play').textContent='▶ Play'; }
    document.getElementById('scrub').value=t/(NA()-1); }
  document.getElementById('frame').textContent=frameLabel();
  draw(); drawChart();
  requestAnimationFrame(tick);
}

document.getElementById('play').onclick=function(){ playing=!playing;
  if(playing&&t>=NA()-1)t=0; this.textContent=playing?'❚❚ Pause':'▶ Play'; };
document.getElementById('scrub').oninput=function(){ t=(+this.value)*(NA()-1); playing=false;
  document.getElementById('play').textContent='▶ Play'; };
cv.style.cursor='grab';
cv.addEventListener('mousedown',ev=>{
  vdrag={x:ev.clientX,y:ev.clientY,vx,vy};
  cv.style.cursor='grabbing';
});
window.addEventListener('mouseup',()=>{ vdrag=null; cv.style.cursor='grab'; });
cv.addEventListener('wheel',ev=>{
  ev.preventDefault();
  const r=cv.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top;
  const ns=Math.min(14,Math.max(0.4,vs*Math.exp(-ev.deltaY*0.0022))), g=ns/vs;
  vx=mx-(mx-vx)*g; vy=my-(my-vy)*g; vs=ns;   // zoom about the cursor
  tip.style.opacity=0;
},{passive:false});
cv.addEventListener('dblclick',()=>{ vs=1; vx=vy=0; });
cv.addEventListener('mousemove',ev=>{
  if(vdrag){
    vx=vdrag.vx+(ev.clientX-vdrag.x); vy=vdrag.vy+(ev.clientY-vdrag.y);
    tip.style.opacity=0; return;
  }
  const r=cv.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top;
  const P=px(); const {inS,outS}=ST||compute(); const md=maxDeg();
  let best=-1,bd=1e9;
  for(let i=0;i<D.nodes.length;i++){ const rr=5+27*Math.sqrt(Math.min(1,(inS[i]+outS[i])/md));
    const d=Math.hypot(P[i].x-mx,P[i].y-my); if(d<rr+6&&d<bd){bd=d;best=i;} }
  hover=best;
  if(best>=0&&ST){
    const {inS,outS,E,pr}=ST;
    const rows=[];
    for(const e of E){ let o=-1; if(e.a===best){o=e.b;} else if(e.b===best){o=e.a;} else continue;
      if(e.ud>0.008) rows.push([D.nodes[o].label,e.ud]); }
    rows.sort((a,b)=>b[1]-a[1]);
    const crank=[...Array(D.nodes.length).keys()].sort((a,b)=>pr[b]-pr[a]).indexOf(best)+1;
    let h='<b>'+D.nodes[best].label+'</b> · '+frameLabel();
    h+='<div class="meta">influence #'+crank+' · draws-in '+inS[best].toFixed(2)+' · reaches-out '+outS[best].toFixed(2)+'</div>';
    if(!rows.length) h+='<span class="r">no active ties</span>';
    for(const [lab,w] of rows.slice(0,8)) h+='<div class="r"><span>'+lab+'</span><span>'+w.toFixed(2)+'</span></div>';
    tip.innerHTML=h; tip.style.opacity=1;
    let tx=mx+14,ty=my+14; if(tx>W-stocksW()-300)tx=mx-tip.offsetWidth-14;
    tip.style.left=tx+'px'; tip.style.top=ty+'px';
  } else tip.style.opacity=0;
});
cv.addEventListener('mouseleave',()=>{hover=-1;tip.style.opacity=0;});
resize(); requestAnimationFrame(tick);
</script></body></html>"""


if __name__ == "__main__":
    main()
