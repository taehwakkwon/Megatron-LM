#!/bin/bash
# End-to-end determinism of one variant on one node with 8 GPUs: --deterministic-mode, off vs on,
# lm loss and grad norm must be bitwise equal every iteration (analyze.py compares them).
# Usage: OUT_ROOT=/shared/e19 bash exp_aux_defer_2p24/run_e2e.sh <variant> [iters]
# Configs: ep8 (TP1 EP8 PP1, MTP 1 layer) and pp2vpp2_rc (TP2+SP EP4 PP2 VPP2, full recompute).
set -uo pipefail
VARIANT=${1:?variant}; ITERS=${2:-30}
export PYTHONPATH=$PWD:${PYTHONPATH:-} NCCL_ALGO=Ring NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
BASE=(
  --num-layers 8 --hidden-size 1024 --num-attention-heads 16 --ffn-hidden-size 4096
  --seq-length 2048 --max-position-embeddings 2048
  --position-embedding-type rope --swiglu --normalization RMSNorm
  --untie-embeddings-and-output-weights --disable-bias-linear
  --num-experts 64 --moe-router-topk 6 --moe-ffn-hidden-size 512
  --moe-router-load-balancing-type global_aux_loss --moe-aux-loss-coeff 1e-2
  --moe-token-dispatcher-type alltoall --moe-grouped-gemm
  --micro-batch-size 1 --global-batch-size 64
  --train-iters $ITERS --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine --lr-warmup-iters 5
  --weight-decay 0.1 --clip-grad 1.0 --bf16 --seed 1234 --use-distributed-optimizer
  --eval-iters 0 --eval-interval 100000 --save-interval 1000000
  --log-interval 1 --distributed-backend nccl
  --mock-data --tokenizer-type NullTokenizer --vocab-size 32768 --split 99,1,0
  --deterministic-mode --no-gradient-accumulation-fusion
)
declare -A CFG=(
  [ep8]="--tensor-model-parallel-size 1 --expert-model-parallel-size 8 --pipeline-model-parallel-size 1 --mtp-num-layers 1"
  [pp2vpp2_rc]="--tensor-model-parallel-size 2 --sequence-parallel --expert-model-parallel-size 4 --pipeline-model-parallel-size 2 --num-layers-per-virtual-pipeline-stage 2 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
)
rc=0
for c in ep8 pp2vpp2_rc; do
  for side in off on; do
    RUN=${VARIANT}_${c}_${side}; OUT=${OUT_ROOT:?set OUT_ROOT}/e2e/$RUN; mkdir -p "$OUT"
    EXTRA=(${CFG[$c]} --tensorboard-dir $OUT/tb --tensorboard-log-interval 1)
    [ "$side" = on ] && EXTRA+=(--moe-router-defer-global-aux-loss)
    if [[ "${CFG[$c]}" == *tensor-model-parallel-size\ 2* ]]; then export CUDA_DEVICE_MAX_CONNECTIONS=1; else unset CUDA_DEVICE_MAX_CONNECTIONS; fi
    echo "===== E2E $RUN head=$(git rev-parse HEAD) ====="
    python -m torch.distributed.run --nproc-per-node 8 --master-port ${MASTER_PORT:-29542} \
      exp_aux_defer_2p24/counters.py pretrain_gpt.py "${BASE[@]}" "${EXTRA[@]}" > "$OUT/node0.log" 2>&1 \
      || { echo "E2E_FAILED $RUN"; rc=1; }
    grep -m1 "AUX_COUNT" "$OUT/node0.log" || true
  done
done
echo "E2E_DONE rc=$rc"
