#!/bin/bash
# Build docs/physics/ffno-physics.pdf and docs/technical/ffno-technical.pdf (pdflatex, three passes for the contents and references).
set -e
cd "$(dirname "$0")"
for t in physics/ffno-physics technical/ffno-technical; do
    d=$(dirname "$t"); b=$(basename "$t")
    (cd "$d" && for pass in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error "$b.tex" > /dev/null; done) || { echo "FAILED: $t"; tail -30 "$t.log"; exit 1; }
    echo "built $t.pdf"
done
rm -f */*.aux */*.log */*.out */*.toc
