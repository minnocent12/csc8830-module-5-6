#!/usr/bin/env bash
# Build the final Module 5-6 PDF report from its Markdown source.
#
# Pipeline: FINAL_REPORT.md --(pandoc, MathML math)--> FINAL_REPORT.html
#           --(headless Chrome, print-to-pdf)--> FINAL_REPORT.pdf
#
# No LaTeX toolchain or internet access is required: pandoc emits native MathML (which Chrome
# renders without JavaScript), and headless Chrome prints that HTML straight to PDF.
#
# Usage: scripts run from anywhere; paths below are relative to this script's own directory.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

pandoc "$DIR/FINAL_REPORT.md" \
    -o "$DIR/FINAL_REPORT.html" \
    --standalone \
    --math-method=mathml \
    --css="report.css" \
    --include-in-header="$DIR/head_include.html"

"$CHROME" \
    --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --print-to-pdf="$DIR/FINAL_REPORT.pdf" \
    "file://$DIR/FINAL_REPORT.html"

rm -f "$DIR/FINAL_REPORT.html"
echo "Built: $DIR/FINAL_REPORT.pdf"
