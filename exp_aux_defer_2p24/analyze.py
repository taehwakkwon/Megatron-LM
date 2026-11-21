"""Summarize E19 results under OUT_ROOT and apply the README decision rule.
Usage: python analyze.py OUT_ROOT"""
import glob, os, re, statistics, sys
from collections import defaultdict

root = sys.argv[1]
VARIANTS = "ABCDE"
DIFF = {"A": 0, "B": 80, "C": 5, "D": 78, "E": 84}  # production lines changed (router.py)

# 1) Unit tests.
unit = {}
for v in VARIANTS:
    p = f"{root}/unit_{v}/unit.log"
    if not os.path.exists(p):
        continue
    text = open(p).read()
    res = {os.path.basename(t): r for r, t in re.findall(r"RESULT (PASS|FAIL) (\S+)", text)}
    new = {n: ("PASS" if re.search(rf"PASSED \S*{n}", text) else "FAIL" if re.search(rf"FAILED \S*{n}", text) else "?")
           for n in ("test_matches_synchronous_above_fp32_integer_range", "test_mixed_synchronous_calls_in_one_step")}
    unit[v] = (res, new)
print("== unit ==")
for v, (res, new) in unit.items():
    print(v, " ".join(f"{k}={r}" for k, r in res.items()), "|", " ".join(f"{k[:30]}={r}" for k, r in new.items()))

# 2) End-to-end determinism: off vs on, lm loss and grad norm per iteration.
def scalars(run_dir):
    tb = glob.glob(f"{run_dir}/tb/events*")
    if tb:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(os.path.dirname(tb[0]), size_guidance={"scalars": 0}); ea.Reload()
            return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in ("lm loss", "grad-norm") if t in ea.Tags()["scalars"]}, "tensorboard"
        except Exception as e:  # fall back to the printed values
            print("tensorboard read failed:", e)
    log = open(f"{run_dir}/node0.log").read()
    it = re.findall(r"iteration\s+(\d+)/.*?lm loss: (\S+) .*?grad norm: (\S+) ", log)
    return {"lm loss": [(int(i), l) for i, l, _ in it], "grad-norm": [(int(i), g) for i, _, g in it]}, "log (printed precision)"

e2e = defaultdict(dict)
print("== e2e (off vs on bitwise) ==")
for off in sorted(glob.glob(f"{root}/e2e/*_off")):
    on = off[:-4] + "_on"
    name = os.path.basename(off)[:-4]
    v = name.split("_")[0]
    if not os.path.exists(f"{on}/node0.log"):
        continue
    a, src = scalars(off); b, _ = scalars(on)
    n = len(a.get("lm loss", []))
    ok = n > 0 and all(a[t] == b.get(t) for t in a)
    first = next((x[0] for t in a for x, y in zip(a[t], b.get(t, [])) if x != y), None)
    e2e[v][name] = ok
    print(f"{name}: iters={n} equal={ok} first_diff_iter={first} source={src}")

# 3) Throughput: median TFLOP/s/GPU over iterations 11-50, per run.
perf = defaultdict(dict)
for d in sorted(glob.glob(f"{root}/perf/*")):
    log = os.path.join(d, "node0.log")
    if not os.path.exists(log):
        continue
    vals = [float(t) for i, t in re.findall(r"iteration\s+(\d+)/.*?TFLOP/s/GPU\): ([\d.]+)", open(log).read()) if 11 <= int(i) <= 50]
    folds = re.findall(r"fold_total=(\d+) fold_ready=(\d+)", open(log).read())
    v, rest = os.path.basename(d).split("_", 1)
    cfg, side, rep = rest.rsplit("_", 2)
    key = "off" if side == "off" else v
    perf[(cfg, rep)][key] = statistics.median(vals) if len(vals) >= 30 else None
    if folds and side == "on":
        t = sum(int(a) for a, _ in folds); r = sum(int(b) for _, b in folds)
        print(f"fold readiness {os.path.basename(d)}: {r}/{t}")
print("== perf: median TFLOP/s/GPU (iters 11-50) ==")
keys = ["off"] + list(VARIANTS)
print("config,rep," + ",".join(keys))
for (cfg, rep), row in sorted(perf.items()):
    print(f"{cfg},{rep}," + ",".join(f"{row[k]:.2f}" if row.get(k) else "-" for k in keys))

# 4) Decision rule.
print("== decision ==")
passing = []
for v in VARIANTS:
    res, new = unit.get(v, ({}, {}))
    gate1 = bool(res) and all(r == "PASS" for r in res.values()) and all(r == "PASS" for r in new.values()) \
        and bool(e2e.get(v)) and all(e2e[v].values())
    gate2 = True
    for row in perf.values():
        a, x, off = row.get("A"), row.get(v), row.get("off")
        if a is None or x is None:
            gate2 = None if gate2 else gate2; continue
        if x < 0.99 * a or (off and a >= 1.01 * off and x < 1.01 * off):
            gate2 = False
    print(f"{v}: correctness={'PASS' if gate1 else 'FAIL'} perf={ {True: 'PASS', False: 'FAIL', None: 'MISSING'}[gate2] } diff={DIFF[v]}")
    if gate1 and gate2:
        passing.append(v)
print("pick:", min(passing, key=DIFF.get) if passing else "none passes both gates -> keep A and qualify the claims")
