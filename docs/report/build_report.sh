#!/usr/bin/env bash
# Build both the final Module 5-6 report deliverables from the single Markdown source.
#
# PDF pipeline (the official, professor-required submission copy):
#   FINAL_REPORT.md --(pandoc, MathML math, head_include.html for the <title> tag only)-->
#   FINAL_REPORT.html --(headless Chrome, print-to-pdf)--> FINAL_REPORT.pdf
#
# DOCX pipeline (an editable companion copy):
#   FINAL_REPORT.md --(pandoc, MathML math, blank title to suppress docx's auto title block)-->
#   an intermediate HTML --(pandoc -f html -t docx)--> FINAL_REPORT.docx
#
# The DOCX is built from HTML, not directly from the Markdown, because pandoc's
# Markdown-to-docx writer does not process raw HTML <figure>/<img> blocks (this report's
# figures are raw HTML, needed for the figure-row/caption layout the PDF uses) - it silently
# drops them. Reading the already-rendered HTML instead (where pandoc's own HTML *reader*
# parses the <figure>/<img>/<math> elements properly) preserves every image, caption, and
# equation. Image <img> tags in the Markdown carry explicit width="N" attributes (one per
# image, chosen from that image's real aspect ratio) specifically for the DOCX path: without
# them, Word embeds images at native pixel size, which is far larger than a page for these
# real photos/video frames and forces one image per page with its caption orphaned onto the
# next page. The CSS max-height rule already constrains the PDF path regardless of these
# attributes, so they are safe for both outputs. Likewise, equations in the real SfM
# mathematical workout (Section 12) are written as separate single-line $$...$$ blocks rather
# than a LaTeX `aligned` block: `aligned` renders fine in the PDF's native MathML, but survives
# the MathML-to-OMML conversion for DOCX badly (literal "&" characters and page-width overflow
# with no line wrapping) - plain consecutive display equations work cleanly in both.
#
# No LaTeX toolchain or internet access is required for either path: pandoc emits native
# MathML/OMML math directly, Chrome prints HTML straight to PDF, and pandoc's own HTML<->docx
# conversion needs nothing external.
#
# Usage: scripts run from anywhere; paths below are relative to this script's own directory.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# Both pandoc conversions below resolve each <img src="../../results/..."> relative to the
# *current working directory*, not relative to the input file's own location - so this script
# must run from its own directory regardless of where it was invoked from, or every image
# silently fails to embed in the DOCX (pandoc replaces it with its alt-text description instead
# of raising an error).
cd "$DIR"

# --- PDF ---
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

# --- DOCX ---
pandoc "$DIR/FINAL_REPORT.md" \
    -o "$DIR/FINAL_REPORT_tmp.html" \
    --standalone \
    --math-method=mathml \
    --css="report.css" \
    --metadata title=" "

pandoc -f html -t docx "$DIR/FINAL_REPORT_tmp.html" -o "$DIR/FINAL_REPORT.docx"

rm -f "$DIR/FINAL_REPORT_tmp.html"
echo "Built: $DIR/FINAL_REPORT.docx"
