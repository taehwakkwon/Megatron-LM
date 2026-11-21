#!/bin/bash
# One throughput run; start once per node (e.g. srun --ntasks-per-node=1) from the variant's repo root.
# Usage: OUT_ROOT=... NNODES=.. NODE_RANK=.. MASTER_ADDR=.. MASTER_PORT=.. \
#        bash exp_aux_defer_2p24/run_perf.sh <variant> <config> <defer:0|1> <repeat> [iters]
# config: 4b_ep1_tp1 | 8b_ep2_tp1 | 8b_ep2_tp2. Run name: <variant>_<config>_<off|on>_r<repeat>.
set -euo pipefail
VARIANT=$1; CONFIG=$2; DEFER=$3; REP=$4; ITERS=${5:-50}
NNODES=${NNODES:-${SLURM_NNODES:-1}}; NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
MASTER_ADDR=${MASTER_ADDR:-localhost}; MASTER_PORT=${MASTER_PORT:-29541}
case $CONFIG in
  4b_ep1_tp1) EXPERTS=32; EP=1; TP=1 ;;
  8b_ep2_tp1) EXPERTS=64; EP=2; TP=1 ;;
  8b_ep2_tp2) EXPERTS=64; EP=2; TP=2 ;;
  *) echo "bad config $CONFIG"; exit 1 ;;
esac
SIDE=$([ "$DEFER" = 1 ] && echo on || echo off)
RUN=${VARIANT}_${CONFIG}_${SIDE}_r${REP}
OUT=${OUT_ROOT:?set OUT_ROOT}/perf/$RUN; mkdir -p "$OUT"
export PYTHONPATH=$PWD:${PYTHONPATH:-} NCCL_DEBUG=${NCCL_DEBUG:-WARN}
ARGS=(
  --num-layers 24 --hidden-size 2048 --num-attention-heads 16 --group-query-attention --num-query-groups 4
  --ffn-hidden-size 8192 --num-experts $EXPERTS --moe-router-topk 4 --moe-ffn-hidden-size 768
  --seq-length 8192 --max-position-embeddings 8192
  --position-embedding-type rope --swiglu --normalization RMSNorm
  --untie-embeddings-and-output-weights --disable-bias-linear
  --moe-router-load-balancing-type global_aux_loss --moe-aux-loss-coeff 1e-3
  --moe-token-dispatcher-type alltoall --moe-grouped-gemm --moe-permute-fusion --moe-router-fusion
  --cross-entropy-loss-fusion --cross-entropy-fusion-impl te
  --expert-model-parallel-size $EP --tensor-model-parallel-size $TP --pipeline-model-parallel-size 1
  --micro-batch-size 2 --global-batch-size 256
  --train-iters $ITERS --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine --lr-warmup-iters 5
  --weight-decay 0.1 --clip-grad 1.0 --bf16 --seed 1234
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather --manual-gc --manual-gc-interval 10
  --eval-iters 0 --eval-interval 100000 --save-interval 1000000
  --log-interval 1 --log-throughput --distributed-backend nccl
  --mock-data --tokenizer-type NullTokenizer --vocab-size 65536 --split 99,1,0
)
[ "$TP" -gt 1 ] && { ARGS+=(--sequence-parallel); export CUDA_DEVICE_MAX_CONNECTIONS=1; }
[ "$DEFER" = 1 ] && ARGS+=(--moe-router-defer-global-aux-loss)
echo "RUN $RUN head=$(git rev-parse HEAD) dirty=$(git status --short | wc -l) nnodes=$NNODES node_rank=$NODE_RANK cdmc=${CUDA_DEVICE_MAX_CONNECTIONS:-unset}"
python -m torch.distributed.run --nnodes $NNODES --nproc-per-node 8 --node-rank $NODE_RANK \
  --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
  exp_aux_defer_2p24/counters.py pretrain_gpt.py "${ARGS[@]}" 2>&1 | tee "$OUT/node${NODE_RANK}.log"
