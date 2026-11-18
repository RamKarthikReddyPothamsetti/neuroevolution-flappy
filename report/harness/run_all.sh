#!/usr/bin/env bash
# Reproduce everything. Usage: bash run_all.sh /path/to/flappy.py WORKDIR REPORTDIR
# ~20 min wall time on 2 cores (exp 1+2: ~5 min, sweep: ~11 min, held-out: ~1.5 min, GIF: ~2.5 min).
set -euo pipefail
ORIG=${1:?path to flappy.py}; WORK=${2:-/tmp/flappy-exp}; REPORT=${3:-./flappy-report}
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK" "$REPORT/figures"
pip install --break-system-packages pygame-ce numpy matplotlib pillow
python "$H/run_experiments.py" "$ORIG" "$WORK" --seeds 20 --only evo,random   # experiments 1 + 2
python "$H/run_experiments.py" "$ORIG" "$WORK" --seeds 20 --only sweep        # experiment 3
python "$H/heldout_test.py"    "$ORIG" "$WORK" --layouts 20                   # experiment 4
python "$H/analyze.py" "$WORK" "$REPORT"                                      # figures, results.json, results.md
python "$H/make_gif.py" "$ORIG" "$REPORT/figures/demo.gif"                    # optional demo
