#!/usr/bin/env python3
"""Canonical-vocabulary merge pass (v2 — cluster then adjudicate).

Stage A (deterministic): union-find candidate clusters over all slugs via
  - normalized-slug equality (alnum only)
  - shared alias surface forms
  - shared display strings
  - slug prefix/containment (teleport / teleport-router)
Stage B (Opus 4.8): adjudicate ONLY multi-member clusters -> merge groups.

Writes data/canonical_vocab.json:
  {"canon": {variant_slug: canonical_id}, "meta": {canonical_id: {display, type}}}
"""
import collections
import csv
import glob
import json
import os
import instance_config as _ic
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-opus-4-8"
IN_PRICE, OUT_PRICE = 5.00 / 1e6, 25.00 / 1e6


def build_inventory():
    inv = {}
    for p in glob.glob(os.path.join(HERE, "data", "subjects", "*.json")):
        d = json.load(open(p))
        for s in d.get("subjects", []):
            e = inv.setdefault(s["id"], {"types": collections.Counter(),
                                         "displays": collections.Counter(),
                                         "aliases": set(), "mentions": 0, "transcripts": 0})
            e["types"][s["type"]] += 1
            e["displays"][s["display"]] += 1
            e["aliases"].update(a.lower().strip() for a in s.get("aliases", []) if len(a) > 2)
            e["mentions"] += len(s.get("mentions", []))
            e["transcripts"] += 1
    return inv


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


GENERIC_ALIASES = {"the", "and", "team", "project", "app", "tool", "agent", "agents",
                   "system", "platform", "product", "demo", "data", "model", "ai"}


def candidate_clusters(inv):
    uf = UF()
    by_norm = collections.defaultdict(list)
    by_alias = collections.defaultdict(list)
    by_disp = collections.defaultdict(list)
    for slug, e in inv.items():
        by_norm[norm(slug)].append(slug)
        for a in e["aliases"]:
            na = norm(a)
            if na and na not in GENERIC_ALIASES and len(na) > 3:
                by_alias[na].append(slug)
        for d in e["displays"]:
            by_disp[norm(d)].append(slug)
    for group in list(by_norm.values()) + list(by_disp.values()):
        for s in group[1:]:
            uf.union(group[0], s)
    for group in by_alias.values():
        if len(group) <= 6:  # huge alias fan-outs are noise (e.g. "tee")
            for s in group[1:]:
                uf.union(group[0], s)
    # slug containment for slugs sharing a first token
    slugs = sorted(inv)
    by_first = collections.defaultdict(list)
    for s in slugs:
        by_first[s.split("-")[0]].append(s)
    for group in by_first.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a in b or b in a:
                    uf.union(a, b)
    clusters = collections.defaultdict(list)
    for s in inv:
        clusters[uf.find(s)].append(s)
    return [sorted(v) for v in clusters.values() if len(v) > 1]


PROMPT = _ic.fill_program("""You are canonicalizing the subject vocabulary extracted from the meeting transcripts of the {PROGRAM} (spring 2026). Transcripts were processed independently, so the same entity appears under multiple slugs (spelling variants, ASR garbles, abbreviation vs full name).

Below are CANDIDATE clusters produced by string/alias matching. For each cluster, decide which slugs (if any) refer to THE SAME real-world entity and should merge. A cluster may split into multiple groups or dissolve entirely — string similarity is NOT sameness (e.g. `agent-economy` vs `agent-memory` must NOT merge; a team and its product are distinct unless the corpus uses them interchangeably).

Each candidate line: `slug | type | display | tx=<transcripts> m=<mentions> | aliases: ...`

{clusters}

## Output
Respond with ONLY JSON (no fences). Include ONLY groups of 2+ slugs you are confident about:
{{"merges": [{{"canonical_id": "...", "display": "...", "type": "...", "variants": ["slug-a", "slug-b"]}}]}}
Pick the most standard slug as canonical_id; `variants` lists all merged slugs including the canonical.""")


def main():
    inv = build_inventory()
    clusters = candidate_clusters(inv)
    n_in_clusters = sum(len(c) for c in clusters)
    print(f"inventory: {len(inv)} slugs; candidate clusters: {len(clusters)} covering {n_in_clusters} slugs")

    blocks = []
    for i, c in enumerate(clusters):
        lines = []
        for slug in c:
            e = inv[slug]
            typ = e["types"].most_common(1)[0][0]
            disp = e["displays"].most_common(1)[0][0]
            al = "; ".join(sorted(e["aliases"]))[:150]
            lines.append(f"  {slug} | {typ} | {disp} | tx={e['transcripts']} m={e['mentions']} | aliases: {al}")
        blocks.append(f"cluster {i}:\n" + "\n".join(lines))
    ctext = "\n\n".join(blocks)
    print(f"cluster text ~{len(ctext)//4} tokens")

    with open(os.path.join(HERE, ".env")) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)
    import anthropic
    client = anthropic.Anthropic(max_retries=3)

    BATCH = 40  # clusters per call — small enough that the merges output never truncates
    merges = []

    def run_batch(block_slice, mtok=48000):
        chunk = "\n\n".join(block_slice)
        with client.messages.stream(
            model=MODEL, max_tokens=mtok,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": PROMPT.format(clusters=chunk)}],
        ) as stream:
            msg = stream.get_final_message()
        u = msg.usage
        cost = u.input_tokens * IN_PRICE + u.output_tokens * OUT_PRICE
        with open(os.path.join(HERE, "data", "usage_log.csv"), "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), "ALL", "vocab-merge",
                                    MODEL, u.input_tokens, u.cache_read_input_tokens or 0,
                                    u.output_tokens, f"{cost:.4f}"])
        text_out = next((b.text for b in msg.content if b.type == "text"), "")
        if not text_out:
            raise RuntimeError(f"no text block; stop_reason={msg.stop_reason}")
        text_out = re.sub(r"^```(?:json)?\s*|\s*```$", "", text_out.strip())
        return json.loads(text_out[text_out.find("{"):text_out.rfind("}") + 1])["merges"], cost

    for bi in range(0, len(blocks), BATCH):
        sl = blocks[bi:bi + BATCH]
        try:
            m, cost = run_batch(sl)
        except (json.JSONDecodeError, KeyError):   # truncated/garbled → split the batch and retry
            m, cost = [], 0.0
            for half in (sl[:len(sl) // 2], sl[len(sl) // 2:]):
                if not half:
                    continue
                hm, hc = run_batch(half)
                m += hm; cost += hc
        merges.extend(m)
        print(f"batch {bi // BATCH}: {len(m)} groups, ${cost:.2f}")

    data = {"merges": merges}
    canon, meta = {}, {}
    for g in data["merges"]:
        cid = g["canonical_id"]
        meta[cid] = {"display": g["display"], "type": g["type"]}
        for v in g["variants"]:
            if v in inv:
                canon[v] = cid
    for slug, e in inv.items():
        if slug not in canon:
            canon[slug] = slug
            meta.setdefault(slug, {"display": e["displays"].most_common(1)[0][0],
                                   "type": e["types"].most_common(1)[0][0]})
    out = {"canon": canon, "meta": meta, "n_groups": len(data["merges"])}
    with open(os.path.join(HERE, "data", "canonical_vocab.json"), "w") as f:
        json.dump(out, f, indent=1)
    merged_away = sum(1 for k, v in canon.items() if k != v)
    print(f"merge groups: {len(data['merges'])}, slugs merged away: {merged_away}, "
          f"canonical subjects: {len(set(canon.values()))}")


if __name__ == "__main__":
    main()
