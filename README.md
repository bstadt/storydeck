# Storydeck

Turn any vault of meeting transcripts into an explorable, grounded story system:

- **stories** — a 3D deck of every conversation a subject appears in; each
  subject gets a written arc narrative grounded in mined evidence, zoomable
  from arc → conversation → transcript span.
- **geometry** — the cohort's collaboration graph over time: influence /
  centrality rankings, model-authored weekly working-group partitions, pan/zoom.
- **query** — a chat oracle over the whole mined corpus with triple-level
  citations (hover a chip to see the exact knowledge-graph triple), plus a
  **compare mode** that runs a time-cutoff "historic" oracle against the
  full-corpus oracle and a judge that synthesizes where they diverge.

Everything the models say is grounded in a mined layer of per-meeting
summaries and (subject, relation, object) triples with provenance.

## Quickstart (synthetic demo corpus)

```bash
pip install flask anthropic
cp .env.example .env            # add your ANTHROPIC_API_KEY
export STORYDECK_INSTANCE=examples/demo-instance.json
python3 storydeck.py ingest examples/demo-vault
python3 storydeck.py mine       # paid stages; prints estimates and asks first
python3 storydeck.py build
python3 storydeck.py serve      # → http://localhost:8811/
```

## Bring your own corpus

1. **Transcripts**: a folder of `.md`/`.txt` files, one per meeting. Optional
   `.json` sidecar per file: `{"date", "title", "meeting_type", "participants"}`
   (else date/title are parsed from a `YYYY-MM-DD-title` filename).
2. **Instance config**: copy `examples/demo-instance.json`, set your program's
   framing phrases and the subject roster (`stems` are the substrings/ASR
   variants that identify a subject; `exclude` guards false matches). Point
   `STORYDECK_INSTANCE` at it — the repo's `instance.json` is the default.
3. `storydeck.py ingest <folder>` normalizes into `data/vault/` (the internal
   contract: `index.csv` + `transcripts/`), then `mine` → `build` → `serve`.

Mining cost scales with corpus size (~$0.12/transcript for facts, plus
relations / weekly partitions / stories). Every paid stage is resumable and
asks before spending; set `STORYDECK_YES=1` to skip prompts in automation.

## Architecture

```
ingest_local.py | sync_transcripts.py     adapters -> data/vault/ (index.csv + transcripts/)
extract_runner.py                         transcript -> subjects + narrative beats
extract_facts.py                          per-meeting summary + grounded triples (resumable)
gen_cohort_relations.py                   per-subject directed, ranked relation estimates over time
gen_week_groups.py                        model-authored weekly working-group partitions
gen_community_labels.py                   community labels/descriptions
gen_story_triplets.py / gen_narratives.py subject arc stories
build_cohort_viz.py                       -> viewer/geometry.html (self-contained)
pipeline.py / ingest.py                   registry, activation, showcase data, S3 keep-current
serve.py                                  Flask: viewer + /api/query (+ compare & judge) + activation
instance_config.py / instance.json        everything corpus-specific (framing, roster, aliases)
```

Registry + activation (stories tab backend):

- `pipeline.py registry` — rebuild the active/potential subject registry
- `pipeline.py activate <subject>` — activate (fuzzy-resolves names/aliases)
- `pipeline.py corpus` — rebuild the whole corpus layer
- `gen_narratives.py --only <id>` — regenerate one subject's grounded arc
- `python3 ingest.py` — keep-current for the CoordinationOS S3 share: syncs new
  transcripts, refreshes touched active subjects' stories, surfaces new
  potential subjects as ranked suggestions

`data/`, `viewer/data/`, `.env`, and key material are gitignored — code and
corpus stay separate. Confidential meeting content: do not redistribute.
`attic/` holds retired experiments.

## Deploying

The stories/geometry tabs are static once built. `/api/query` holds your API
key and spends real money per question (prompt-cache warms are dollars, not
cents, on large corpora) — put it behind auth and rate limits before exposing
it to the internet. `scripts/export_oss.sh` produces a clean code-only tree
(no corpus data, no built artifacts) for publishing.

## License

MIT — see LICENSE.
