# Experiment E19: how the deferred global aux loss should accumulate counts (PR #9378)

Reviewer follow-up on NVIDIA/Megatron-LM#9378. Above 2**24 cumulative selections per expert in
one optimizer step, the deferred path (exact int64 running sum, one cast) and the synchronous path
(fp32 `F += g` per micro-batch) read different averages. A related failure: if a synchronous call
(for example CUDA-graph warmup) runs inside a step of deferred calls, the int64 running sum of the
deferred calls misses its counts.

Each variant is a branch on the fork `taehwakkwon/Megatron-LM`, based on the PR head 32a83da37.
Every branch carries the same two new tests and this directory.

| Branch `exp/aux-defer-2p24-*` | Variant | What changes |
|---|---|---|
| `A` | current PR | nothing (baseline): prefix all-reduce of 2E int64, exact average |
| `B` | backward cursor | all-reduce E counts; backward adds pending counts in launch order through a cursor and keeps each running average |
| `C` | int64 sync path | `global_tokens_per_expert`/`ga_steps` become int64, so both paths are exact (changes default-path numerics above 2**24) |
| `D` | side stream | forward hands the wait and the fp32 accumulation to a side CUDA stream; backward waits on an event |
| `E` | lag-1 | each forward first adds the previous call's counts (waiting its all-reduce), snapshots the running sum; backward averages `(snapshot + own) / (steps + 1)` |

Production diff (router.py): A 0, B +48/-32, C +3/-2, D +45/-33, E +52/-32.

## Hypotheses and expected outcome

- H1 (correctness): B, D, E match the synchronous path bitwise above 2**24 and with mixed calls.
  A fails both new tests; C passes the 2**24 test and fails the mixed test.
- H2 (performance): B and E keep A's throughput (within 1%). D may lose the gain when
  `CUDA_DEVICE_MAX_CONNECTIONS=1` (TP>1). E's forward wait is free if the previous all-reduce has
  finished (`counters.py` reports `fold_ready`/`fold_total`).

## Decision rule (fixed before running)

1. Correctness gate: `test_matches_synchronous_above_fp32_integer_range` and
   `test_mixed_synchronous_calls_in_one_step` pass; `test_aux_loss.py`, `test_routers.py`,
   `test_transformer_config.py` pass; end-to-end off vs on is bitwise equal (lm loss and grad norm,
   every iteration) in every e2e config.
2. Performance gate: median TFLOP/s/GPU (iterations 11-50) within 1% of A in both repeats of every
   perf config, and at least 1% above off wherever A is.
3. Among variants passing both gates, take the smallest production diff. If none passes 1, keep A
   and qualify the bitwise claims instead.

## Budget (8x H100 nodes)

| Stage | Size | GPU-hours |
|---|---|---|
| unit (`run_unit.sh`) | 5 variants x 1 node x ~20 min | ~13 |
| e2e (`run_e2e.sh`) | 5 variants x 2 configs x off/on x ~5 min, 1 node | ~15 |
| perf (`run_perf.sh`) | 3 configs x (off + 5 variants) x 2 repeats x ~6 min, 4 nodes | ~115 |

Run unit and e2e first; run perf only for variants that pass the correctness gate.

## How to run

Prerequisite: an environment that runs Megatron-LM training (NGC PyTorch container with
TransformerEngine). The scripts use `--mock-data`, so no dataset or tokenizer is needed.

```bash
# Check out every variant next to each other.
git clone https://github.com/taehwakkwon/Megatron-LM.git m && cd m
for v in A B C D E; do git worktree add ../v$v origin/exp/aux-defer-2p24-$v; done
export OUT_ROOT=/shared/path/e19          # shared by all nodes and variants

# 1) Unit tests, one node with 8 GPUs, per variant.
for v in A B C D E; do (cd ../v$v && bash exp_aux_defer_2p24/run_unit.sh $v); done

# 2) End-to-end determinism, one node with 8 GPUs, per variant.
for v in A B C D E; do (cd ../v$v && bash exp_aux_defer_2p24/run_e2e.sh $v); done

# 3) Throughput. run_perf.sh runs on every node (one task per node, e.g. under srun) with
#    NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT set. Interleave variants within each repeat.
#    See slurm_example.sbatch for a Slurm loop over configs, repeats and variants.

# 4) Summary table and decision.
python exp_aux_defer_2p24/analyze.py $OUT_ROOT
```

Perf configs (same model as the PR's throughput table: 24 layers, hidden 2048, seq 8192, GBS 256,
topk 4): `4b_ep1_tp1` (32 experts), `8b_ep2_tp1` (64 experts), `8b_ep2_tp2` (64 experts, TP2+SP,
`CUDA_DEVICE_MAX_CONNECTIONS=1`). Run them on 4 nodes to compare with the PR's numbers.
