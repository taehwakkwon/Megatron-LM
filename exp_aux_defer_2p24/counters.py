"""Run a Megatron entry script and count which global aux loss path ran.
Usage: python counters.py pretrain_gpt.py <args...>. Prints one AUX_COUNT line per rank at exit.
calls/deferred: finish_deferred_global_aux_loss calls / calls with a pending all-reduce;
sync: synchronous global aux losses; fold_total/fold_ready: all-reduces a fold waited for /
of which had already completed (variants B and E only; a fold that waits on an unfinished
all-reduce is where waiting could cost time)."""
import atexit, os, runpy, sys
from megatron.core.transformer.moe import router as R

cnt = dict(calls=0, deferred=0, sync=0, fold_total=0, fold_ready=0)
_finish = R.TopKRouter.finish_deferred_global_aux_loss
def finish(self, *a, **k):
    cnt["calls"] += 1
    cnt["deferred"] += self._pending_global_aux_loss is not None
    return _finish(self, *a, **k)
R.TopKRouter.finish_deferred_global_aux_loss = finish
_sync = R.get_tokens_per_expert_and_token_count
def sync(*a, **k):
    cnt["sync"] += 1
    return _sync(*a, **k)
R.get_tokens_per_expert_and_token_count = sync

def _count(works):
    for w in works:
        if w is not None:
            cnt["fold_total"] += 1
            cnt["fold_ready"] += bool(w.is_completed())

if hasattr(R.TopKRouter, "_fold_lagged_global_counts"):  # variant E
    _fold = R.TopKRouter._fold_lagged_global_counts
    def fold(self):
        if self._lagged_global_counts is not None:
            _count([self._lagged_global_counts[0]])
        return _fold(self)
    R.TopKRouter._fold_lagged_global_counts = fold
if hasattr(R.TopKRouter, "_fold_deferred_global_counts"):  # variant B
    _foldb = R.TopKRouter._fold_deferred_global_counts
    def foldb(self, index):
        _count(e[0] for e in self._deferred_global_counts[self._num_folded_global_counts : index + 1])
        return _foldb(self, index)
    R.TopKRouter._fold_deferred_global_counts = foldb

atexit.register(lambda: print("AUX_COUNT rank={} ".format(os.environ.get("RANK"))
                              + " ".join(f"{k}={v}" for k, v in cnt.items()), flush=True))
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
