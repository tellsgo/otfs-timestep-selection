#!/bin/bash
# One-click reproduction of this paper.  Usage (from any directory):
#   bash run_all.sh            # everything: paths, training, evaluation, comparison with the paper
#   bash run_all.sh smoke      # 2-minute environment check (tiny training and evaluation, nothing is kept)
#   bash run_all.sh weights    # download the trained networks of the paper (GitHub release)
#   bash run_all.sh train      # only the networks (resumable, finished networks are skipped)
#   bash run_all.sh eval       # only the evaluations (needs the trained networks)
#   bash run_all.sh check      # compare results/v21 with reference_results (the numbers in the paper)
# Environment: PY (python with torch, default: python), CUDA_VISIBLE_DEVICES (GPU to use).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
export PY="${PY:-python}" ROOT
# The code refers to its own folder by an absolute path.  Point it at this folder (again after the folder is moved).
OLD="__OTFS_ROOT__"; [ -f .root ] && OLD="$(cat .root)"
if [ "$OLD" != "$ROOT" ]; then
  grep -rlF "$OLD" code pipeline | while read -r f; do
    "$PY" - "$f" "$OLD" "$ROOT" <<'PYEOF'
import sys; p, a, b = sys.argv[1:]; s = open(p).read(); open(p, 'w').write(s.replace(a, b))
PYEOF
  done
  echo "$ROOT" > .root
fi
mkdir -p results/v21 runs_v21 lmmse_v21
"$PY" -c "import torch, numpy, scipy" || { echo "PY=$PY cannot import torch/numpy/scipy, see README (环境)"; exit 1; }
STAGE="${1:-all}"
case "$STAGE" in
  smoke)
    # tiny anchors (256 frames instead of 16384, written as *_n256_* and removed again), 20 training steps, 8 test frames
    export OTFS_ANCHOR_DRAWS=256; rm -rf smoke; mkdir -p smoke
    SMOKE=1 "$PY" pipeline/train_run.py $(head -1 pipeline/runs.txt) || { cat smoke/runs_v21/*/train.log | tail -20; exit 1; }
    (cd code && "$PY" -m otfs5.evaluate --out ../smoke/eval.json --methods ls,lmmse --n 8 --snrs 20 --speeds 500 --splits test --anchor_snrs 20 > ../smoke/eval.log 2>&1) || { tail -20 smoke/eval.log; exit 1; }
    grep -i "nmse" smoke/eval.log | tail -3
    rm -rf smoke; find lmmse_v21 -name '*_n256_*' -delete; echo "SMOKE OK" ;;
  weights)
    # trained networks of the paper (GitHub release), so that eval and check can run without training
    T=P1_weights.tar; URL="${WEIGHTS_URL:-https://github.com/tellsgo/otfs-timestep-selection/releases/download/v1.0/$T}"
    [ -f "$T" ] || curl -L --fail -o "$T" "$URL"
    if command -v sha256sum >/dev/null; then sha256sum -c weights.sha256; else shasum -a 256 -c weights.sha256; fi
    tar -xf "$T"; for d in runs_v21/*/; do touch "$d/DONE"; done; rm -f "$T"; echo "WEIGHTS OK" ;;
  train) "$PY" pipeline/train_run.py $(cat pipeline/runs.txt) ;;
  eval)  bash pipeline/steps.sh ;;
  check) "$PY" pipeline/check_results.py ;;
  all)   "$PY" pipeline/train_run.py $(cat pipeline/runs.txt); bash pipeline/steps.sh; "$PY" pipeline/check_results.py ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
