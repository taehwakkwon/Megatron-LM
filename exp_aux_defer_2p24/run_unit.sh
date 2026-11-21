#!/bin/bash
# Unit tests of one variant on one node with 8 GPUs. Run from the variant's repo root.
# Usage: OUT_ROOT=/shared/e19 bash exp_aux_defer_2p24/run_unit.sh <variant>
set -uo pipefail
VARIANT=${1:?variant}
OUT=${OUT_ROOT:?set OUT_ROOT}/unit_$VARIANT; mkdir -p "$OUT"
export PYTHONPATH=$PWD:${PYTHONPATH:-}
{
  echo "VARIANT $VARIANT HEAD $(git rev-parse HEAD) DIRTY $(git status --short | wc -l)"
  rc=0
  for t in tests/unit_tests/transformer/moe/test_aux_loss.py \
           tests/unit_tests/transformer/moe/test_routers.py \
           tests/unit_tests/transformer/test_transformer_config.py; do
    echo "===== PYTEST $t ====="
    python -m torch.distributed.run --nproc-per-node 8 -m pytest -q -p no:cacheprovider -rA \
      -m "not flaky and not flaky_in_dev" "$t" && echo "RESULT PASS $t" || { echo "RESULT FAIL $t"; rc=1; }
  done
  echo "UNIT_DONE rc=$rc"
} 2>&1 | tee "$OUT/unit.log"
