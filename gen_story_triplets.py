#!/usr/bin/env python3
"""Regenerate a subject's story from the corpus TRIPLE graph (new method).

For a subject, gather every grounded triple across the corpus that mentions it,
grouped by source transcript. Pack into context in strict TEMPORAL order (oldest
transcript first; within a transcript, triples ordered by the turn they were
grounded in), each transcript block carrying its metadata (date, title, type)
and summary. Then write a NARRATIVE that emphasizes change/growth/"shape
rotation" of the subject through the program.

Usage: python3 gen_story_triplets.py <subject-key>
"""
import glob
import json
import os
import re
import sys

import extract_facts as ef

import instance_config as _ic

MODEL = "claude-fable-5"

# subject key -> {stems: substrings that identify its node variants, exclude:
# substrings that would be false matches}. Verified against corpus node degrees.
SUBJECTS = _ic.subjects()

ANTI_CLAUDE = """Avoid the generic AI-essay voice completely: NO em-dash clauses stacked for rhythm, NO "not just X but Y" / "wasn't X, it was Y" constructions, NO tricolon build-ups (three parallel phrases), NO tidy uplifting thematic bow at the end, and do not lean on words like "hardened," "crystallized," "settled into," "connective tissue." Commit fully and unmistakably to the target author's voice; a reader should be able to name the author."""

STYLES = {
    "default": "",
    "scott-alexander": """Write in the voice of Scott Alexander (Astral Codex Ten / Slate Star Codex). Think out loud on the page: pose the question you're genuinely curious about and reason toward it. Plain, conversational prose in a first-person analytical voice ("I think", "here's my model", "the interesting thing is"). Explain mechanisms and incentives; treat everyone as an agent responding rationally to their situation. Reach for one vivid concrete analogy when it actually clarifies. Allow tentativeness and steelman the alternative reading before you settle ("Maybe. But probably Y, because…"). Dry understated humor and parenthetical asides are welcome. Curiosity over lyricism.""",
    "vonnegut": """Write in the voice of Kurt Vonnegut. Short, plain, declarative sentences. Small words. Short paragraphs, often one or two sentences. Deadpan, absurdist, darkly funny, but tender toward human folly. State the terrible and the wonderful in the same flat tone. Use a recurring flat refrain if one fits the material. Let irony and fatalism carry the meaning; never explain the moral. "Here is what happened." No fancy words, no elaborate clauses, no uplifting wrap-up.""",
    "isaacson": """Write in the voice of Walter Isaacson, the biographer (of Jobs, Einstein, Franklin, Leonardo, Doudna, Musk). Tell it as narrative biography: build the story out of specific scenes and telling incidents, and let character emerge from what the subject actually does rather than from adjectives you assign. Narrate in detached but attentive third person — clear, authoritative, accessible without being casual. Draw out the defining traits and tensions that drive the subject's trajectory, and situate the subject inside the larger current of the moment (the accelerator, the coordination problem, the AI scene) without losing sight of the individual. Use concrete, well-chosen detail and reconstructed moments. Note flaws alongside strengths, evenhandedly. Do NOT end on an extracted "lesson," moral, or thesis, and do not sum up tidily; let the story rest on a well-chosen final scene or fact.""",
    "de-waal": """Write in the voice of Frans de Waal, the primatologist. Observe this cohort the way an ethologist observes a troop: detached, patient, closely attentive to social behavior — who defers to whom, who forms coalitions, who grooms whom, who is edged to the periphery, how status is won and lost, how conflicts get repaired or don't. Report specific observed episodes in concrete behavioral terms first, then step back to name the general pattern they reveal. Occasionally and lightly, compare a move to what one sees in chimpanzees, bonobos, or other social animals — only where it genuinely illuminates, never forced. Treat the named humans as you would named animals in a long study: track their maneuvering without judging it. Keep the tone cool, curious, humane; neither cynical nor sentimental. Describe, do not moralize. Plain scientific prose, observational rather than lyrical.""",
}

NARRATIVE_PROMPT = _ic.fill_program("""You are writing the STORY of **{subject}** across the {PROGRAM} ({SPAN_ACTIVE}). You work ONLY from mined data below: per-meeting summaries and grounded (subject, relation, object) triples with provenance, presented in STRICT CHRONOLOGICAL ORDER — oldest meeting first, and within each meeting the triples are ordered by when they were said. Each triple reads `subject --relation--> object |cCERTAINTY MODALITY |T:turns`. Certainty is 0-1; modality (asserted/hedged/reported/proposed/planned/hypothetical/negated) tells you how firmly it was put forward.

{context}

## Task
First output a single line beginning `STRAND: ` that compresses {subject}'s whole arc into 4-8 short phrases joined by ` -> `, capturing the transformation (e.g. `STRAND: aspiration -> scrappy prototype -> collapse -> new owner -> shipped doctrine`). Then a blank line, then the story.

Write {subject}'s story as a NARRATIVE whose spine is TRANSFORMATION — the change and growth of {subject} across the program.

STYLE (this matters as much as the content):
{style_directive}
{anti_claude}

Substance requirements (hold these regardless of style):
- Center the ARC OF CHANGE: what {subject} was at the start, how it/they shifted, the turning points, what it/they became by mid-July. Keep moving the transformation forward; do not recap meeting by meeting or list facts in blocks.
- Track how {subject}'s RELATIONSHIPS and role shifted over time.
- Stay grounded in the data; weave dates/meetings in naturally rather than dumping citations. RESPECT CERTAINTY: state asserted facts plainly, but hedge what the data only hedges or proposes.
- Chronological, landing on where {subject}'s trajectory stands by mid-July.

CITATIONS (required): each transcript block above is headed with an index like `[#7]`. End EVERY sentence of the story with a source tag in double square brackets naming the block index(es) that sentence draws from — e.g. `...she hired him.[[7]]` or `...bundled into a lab.[[7,12]]`. Put the tag AFTER the sentence's terminal punctuation. Every sentence gets exactly one tag (cite the single most relevant block, or a few if it genuinely synthesizes across them). The STRAND line gets no tag.

Output ONLY the STRAND line, a blank line, then the story (every sentence tagged).""")


def _minturn(t):
    ns = [int(x) for x in re.findall(r"\d+", str(t.get("evidence_turns") or []))]
    return min(ns) if ns else 10**6


def matches(node, spec):
    n = (node or "").lower()
    if any(x in n for x in spec.get("exclude", [])):
        return False
    return any(s in n for s in spec["stems"])


def build_context(spec):
    tx = {}
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
        matched = [t for t in d["triples"]
                   if matches(t.get("subject"), spec) or matches(t.get("object"), spec)]
        if not matched:
            continue
        matched.sort(key=_minturn)
        m = ef.lookup_meta(tid)
        tx[tid] = {"date": m["date"], "title": m["title"], "mtype": m["mtype"],
                   "summary": d.get("summary", ""), "triples": matched}
    order = sorted(tx, key=lambda t: tx[t]["date"] or "")
    blocks = []
    srcmap = {}   # block index -> transcript id, for sentence-level citations
    for i, tid in enumerate(order):
        e = tx[tid]
        srcmap[i] = tid
        lines = [f"### [#{i}] [{e['date']}] {e['title']} ({e['mtype']})",
                 f"SUMMARY: {e['summary']}"]
        for t in e["triples"]:
            turns = ",".join(str(x) for x in (t.get("evidence_turns") or []))
            lines.append(f"  {t['subject']} --{t['relation']}--> {t['object']} "
                         f"|c{t.get('certainty')} {t.get('modality')} |T:{turns}")
        blocks.append("\n".join(lines))
    ntri = sum(len(tx[t]["triples"]) for t in order)
    return "\n\n".join(blocks), len(order), ntri, srcmap


def main():
    ef.load_env()
    import anthropic
    key = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else "default"
    spec = SUBJECTS[key]
    context, ntx, ntri, srcmap = build_context(spec)
    prompt = NARRATIVE_PROMPT.format(subject=key, context=context,
                                     style_directive=STYLES[style],
                                     anti_claude=ANTI_CLAUDE if style != "default" else "")
    client = anthropic.Anthropic(max_retries=3, timeout=600)
    with client.messages.stream(model=MODEL, max_tokens=8000,
                                messages=[{"role": "user", "content": prompt}]) as st:
        msg = st.get_final_message()
    out = next((b.text for b in msg.content if b.type == "text"), "")
    suffix = "" if style == "default" else f"__{style}"
    dst = f"data/facts/STORY_{key}{suffix}.md"
    open(dst, "w").write(f"<!--SRC {json.dumps(srcmap)}-->\n" + out)
    cost = msg.usage.input_tokens / 1e6 * 10 + msg.usage.output_tokens / 1e6 * 50
    print(f"{key} [{style}]: {ntx} tx · {ntri} tri · in={msg.usage.input_tokens} "
          f"out={msg.usage.output_tokens} ${cost:.2f} -> {dst}")


if __name__ == "__main__":
    main()
