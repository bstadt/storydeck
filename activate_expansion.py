#!/usr/bin/env python3
"""Activate the accelerator-coverage expansion subjects in the viewer: ensure
each has a showcase.json subject entry (synthesized from the triple corpus for
people the old vocab never canonicalized) and an active registry entry with a
color, so they appear in the picker with triplet decks + Isaacson stories."""
import json

import gen_story_triplets as g

import instance_config as _ic

# cid -> (display, type): expansion subjects come from the instance config
NEW = {cid: (v["display"], v["type"])
       for cid, v in _ic.cfg().get("subject_display", {}).items()}
PALETTE = ["#4d94e8", "#f08a56", "#2fd39a", "#e0d24d", "#9ab0ff", "#ef6a68", "#7ee081",
           "#ff8fb0", "#5ac8e0", "#c98bff", "#f0b429", "#8b7ce8", "#3fae3f", "#f291b9",
           "#e86a4d", "#d0d0c8", "#66d9c2", "#dba7f7", "#a3c85a", "#f7a76b", "#6fa8dc",
           "#e8c46a", "#93d1f0", "#f28b82"]


def main():
    show = json.load(open("viewer/data/showcase.json"))
    trip = json.load(open("viewer/data/triples.json"))
    reg = json.load(open("viewer/data/registry.json"))
    for i, (cid, (disp, typ)) in enumerate(NEW.items()):
        spec = g.SUBJECTS[cid]
        n_tx = n_nodes = 0
        for rows in trip.values():
            hit = sum(1 for r in rows if g.matches(r[0], spec) or g.matches(r[2], spec))
            if hit:
                n_tx += 1
                n_nodes += hit
        if cid not in show["subjects"]:
            show["subjects"][cid] = {"display": disp, "type": typ, "n_tx": n_tx,
                                     "hits": [], "voice": []}
        e = reg["subjects"].setdefault(cid, {"display": disp, "type": typ})
        e.update({"n_tx": n_tx, "n_nodes": n_nodes, "status": "active",
                  "color": e.get("color") or PALETTE[i % len(PALETTE)]})
        print(f"  active: {cid:18s} {typ:6s} n_tx={n_tx:3d} mentions={n_nodes}")
    reg["n_active"] = sum(1 for e in reg["subjects"].values() if e.get("status") == "active")
    reg["n_potential"] = sum(1 for e in reg["subjects"].values() if e.get("status") != "active")
    json.dump(show, open("viewer/data/showcase.json", "w"))
    json.dump(reg, open("viewer/data/registry.json", "w"))
    print(f"registry: {reg['n_active']} active, {reg['n_potential']} potential")


if __name__ == "__main__":
    main()
