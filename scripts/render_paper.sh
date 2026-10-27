#!/usr/bin/env bash
# Render docs/PAPER.md -> docs/PAPER.pdf exactly as the Aug 23 / Oct 13 builds
# (pandoc -> HTML5 -> WeasyPrint, A4). Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
pandoc docs/PAPER.md -s -f gfm+smart -t html5 \
  --metadata pagetitle="Coverage, Generalization, and Time" -o "$tmp/paper.html"
weasyprint --base-url "$PWD/docs/" "$tmp/paper.html" docs/PAPER.pdf
rm -rf "$tmp"
pdfinfo docs/PAPER.pdf | grep -E 'Pages|Page size'
