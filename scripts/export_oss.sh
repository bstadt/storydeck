#!/usr/bin/env bash
# Export a clean, publishable Storydeck tree (code only — no corpus data, no
# built artifacts, no instance secrets) into a fresh git repo with a single
# squashed commit. A private instance repo's history contains corpus data
# (built geometry/cohort HTML payloads), so the public repo must NOT share it.
#
# Usage: scripts/export_oss.sh [dest-dir]     (default: ../storydeck-oss)
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$SRC/../storydeck-oss}"

if [ -e "$DEST/.git" ]; then echo "refusing: $DEST already a git repo"; exit 1; fi
mkdir -p "$DEST"

# tracked code, minus corpus-bearing / instance-private files
EXCLUDES=(
  "cohort.html"                 # built, contains instance data payload
  "viewer/geometry.html"        # built, contains instance data payload
  "instance.json"               # private instance binding (roster of real people)
  "POLLING.md"                  # private ops notes
  "viewer/index.html"           # legacy langshare explorer
  "extraction/*"                # legacy prompt docs (superseded by in-code prompts)
  "attic/*"                     # retired experiments
)
cd "$SRC"
git ls-files -z | while IFS= read -r -d '' f; do
  for pat in "${EXCLUDES[@]}"; do
    case "$f" in $pat) continue 2;; esac
  done
  mkdir -p "$DEST/$(dirname "$f")"
  cp "$f" "$DEST/$f"
done

# the demo instance becomes the default instance in the public tree
cp "$SRC/examples/demo-instance.json" "$DEST/instance.json"

cd "$DEST"
git init -q
git add -A
git commit -q -m "Storydeck: grounded story system over a vault of meeting transcripts"
echo "exported to $DEST ($(git ls-files | wc -l | tr -d ' ') files, single squashed commit)"
echo "review it, then: git remote add origin <url> && git push -u origin main"
