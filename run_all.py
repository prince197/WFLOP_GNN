#!/usr/bin/env python3
"""
============================================================
RUN THE WHOLE CAMPAIGN IN THE BACKGROUND
============================================================

Does exactly what submit_gpu.slurm does, but without SLURM, and it
detaches so you get your shell back immediately:

    python run_all.py                 # start it, return at once
    python run_all.py status          # how far along is it?
    python run_all.py log             # follow the output
    python run_all.py stop            # stop it (safely - see below)

Sequence, identical to the SLURM script:

    validate_gpu.py
    for each wind data set:
        run_experiments_gpu.py            (the campaign)
        report.py --paper 1 2 3 4         (four analyses)
    combine_results.py                    (both sets, one workbook)

It survives logout. The child is started in its own session, so
closing the terminal or dropping the SSH connection does not kill it.

STOPPING AND RESUMING IS SAFE
-----------------------------
`stop` lets the current group finish is NOT what happens - it stops the
process, and whatever groups had already completed are in the checkpoint.
Starting again resumes from there, at (radius, turbines, algorithm)
granularity, so nothing already computed is recomputed. That is the same
mechanism a SLURM wall-clock timeout relies on.

ONLY ONE AT A TIME
------------------
Two concurrent runs pointed at the same checkpoint would corrupt it, so
starting a second one while the first is alive is refused. Use `status`
to see what is running, or `stop` first.

WHERE TO RUN IT
---------------
On a compute node (an interactive GPU allocation, e.g.
`salloc --partition=gpu --gres=gpu:a100:1 --cpus-per-task=32`), or on any
machine with the GPU. Do NOT run the full campaign on a login node: it is
hours of compute, login nodes are shared, and most sites kill such jobs.
For a batch queue use submit_gpu.slurm instead - that is what it is for.

OPTIONS
-------
    --datasets 1 2        which wind data sets, in order (default: 1 2)
    --backend gpu|cpu|auto        default: gpu (fails loudly without CUDA)
    --papers 1 2 3 4      which paper analyses to run (default: all four)
    --skip-validation     skip validate_gpu.py (not recommended)
    --budget N            fixed-evaluation regime (WFLOP_BUDGET)
    --smoke               tiny grid, ~2 minutes, to prove the pipeline
============================================================
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "logs", "run_all_state.json")
PIDFILE = os.path.join(HERE, "logs", "run_all.pid")
PY = sys.executable or "python3"


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def read_state():
    try:
        with open(STATE) as fh:
            return json.load(fh)
    except Exception:                            # noqa: BLE001
        return {}


def write_state(**kw):
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    s = read_state()
    s.update(kw)
    s["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(s, fh, indent=2)
    os.replace(tmp, STATE)                       # atomic: status never sees half a file


def alive(pid):
    """Is that pid still running? Signal 0 tests without delivering anything."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def running_pid():
    try:
        with open(PIDFILE) as fh:
            pid = int(fh.read().strip())
    except Exception:                            # noqa: BLE001
        return None
    return pid if alive(pid) else None


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------
def expected_groups(dataset):
    """Groups in one data set's campaign: cases x algorithms.

    Read from the manifest the campaign itself wrote, so this cannot drift
    away from the settings actually in force (WFLOP_CASES, WFLOP_ALGOS and a
    smoke run all change it). Falls back to the full grid before the first
    manifest exists.
    """
    path = os.path.join(HERE, "results", f"manifest_ds{dataset}.json")
    try:
        with open(path) as fh:
            m = json.load(fh)["settings"]
        return int(m["n_cases"]) * len(m["algorithms"])
    except Exception:                            # noqa: BLE001
        return 39 * 14


def done_groups(dataset):
    """Completed groups, counted from the checkpoint the campaign appends to."""
    path = os.path.join(HERE, "results",
                        f"RawResults_checkpoint_ds{dataset}.csv")
    if not os.path.exists(path):
        return 0
    seen = set()
    try:
        with open(path) as fh:
            next(fh, None)                       # header
            for line in fh:
                p = line.split(",", 4)
                if len(p) >= 4:
                    seen.add((p[0], p[1], p[3]))  # radius, turbines, algorithm
    except Exception:                            # noqa: BLE001
        return 0
    return len(seen)


def bar(frac, width=28):
    n = max(0, min(width, int(round(frac * width))))
    return "[" + "#" * n + "." * (width - n) + "]"


# ---------------------------------------------------------------------------
# the worker: this is what actually runs the pipeline
# ---------------------------------------------------------------------------
def worker(args):
    def run(cmd, label):
        print(f"\n$ {' '.join(cmd)}", flush=True)
        return subprocess.call(cmd, cwd=HERE)

    failures = []
    write_state(stage="validation", dataset=None, failures=[])

    if not args.skip_validation:
        print("=" * 74)
        print("VALIDATION")
        print("=" * 74, flush=True)
        if run([PY, "validate_gpu.py"], "validation") != 0:
            print("\nVALIDATION FAILED - stopping before the campaign.\n"
                  "Nothing was run. Fix what it reported, then start again.",
                  flush=True)
            write_state(stage="failed", failures=["validation"])
            return 1

    for ds in args.datasets:
        print("\n" + "=" * 74)
        print(f"WIND DATA SET {ds}")
        print("=" * 74, flush=True)

        write_state(stage="campaign", dataset=ds)
        env_note = f"WFLOP_DATASET={ds}"
        os.environ["WFLOP_DATASET"] = str(ds)
        print(f"\n$ {env_note} {PY} run_experiments_gpu.py", flush=True)
        if subprocess.call([PY, "run_experiments_gpu.py"], cwd=HERE) != 0:
            print(f"CAMPAIGN FAILED for data set {ds} - skipping its reports.",
                  flush=True)
            failures.append(f"campaign_ds{ds}")
            write_state(failures=failures)
            continue

        write_state(stage="reports", dataset=ds)
        for p in args.papers:
            # A failing paper is recorded and does not stop the rest: with a
            # restricted WFLOP_ALGOS some papers legitimately have no data.
            if run([PY, "report.py", "--dataset", str(ds),
                    "--paper", str(p)], f"report ds{ds} p{p}") != 0:
                failures.append(f"report_ds{ds}_paper{p}")
                write_state(failures=failures)

    write_state(stage="combine", dataset=None)
    print("\n" + "=" * 74)
    print("COMBINED RAW RESULTS")
    print("=" * 74, flush=True)
    if run([PY, "combine_results.py", "--datasets",
            *[str(d) for d in args.datasets]], "combine") != 0:
        failures.append("combine")

    write_state(stage="done" if not failures else "done_with_failures",
                failures=failures, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    print("\n" + "=" * 74)
    if failures:
        print(f"FINISHED, with {len(failures)} step(s) that failed:")
        for f in failures:
            print(f"    {f}")
    else:
        print("FINISHED - every step succeeded.")
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 74, flush=True)
    return 0


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_start(args):
    pid = running_pid()
    if pid:
        print(f"A run is already going (pid {pid}).\n"
              "Two runs sharing one checkpoint would corrupt it, so this one "
              "was not started.\n"
              "    python run_all.py status     see how far it is\n"
              "    python run_all.py stop       stop it first")
        return 1

    for d in ("logs", "results", "curves"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = os.path.join(HERE, "logs", f"run_all_{stamp}.log")

    env = dict(os.environ)
    env["WFLOP_BACKEND"] = args.backend
    env.setdefault("WFLOP_DTYPE", "float64")
    env.setdefault("WFLOP_MEM_BUDGET_MB", "2048")
    if args.budget:
        env["WFLOP_BUDGET"] = str(args.budget)
    if args.smoke:
        env["WFLOP_SMOKE"] = "1"

    cmd = [PY, os.path.abspath(__file__), "--_worker",
           "--datasets", *[str(d) for d in args.datasets],
           "--papers", *[str(p) for p in args.papers],
           "--backend", args.backend]
    if args.skip_validation:
        cmd.append("--skip-validation")

    with open(log, "w") as fh:
        # start_new_session detaches the child from this terminal, so it keeps
        # running after logout or an SSH drop.
        proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=fh,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                start_new_session=True)

    with open(PIDFILE, "w") as fh:
        fh.write(str(proc.pid))
    write_state(pid=proc.pid, log=log, stage="starting",
                datasets=args.datasets, papers=args.papers,
                backend=args.backend, failures=[],
                started=time.strftime("%Y-%m-%d %H:%M:%S"))

    print("=" * 70)
    print(f"Started in the background   pid {proc.pid}")
    print(f"backend {args.backend} | data sets {args.datasets} | "
          f"papers {args.papers}")
    print(f"log  {log}")
    print("=" * 70)
    print("It keeps running after you log out. Check on it with:")
    print("    python run_all.py status")
    print("    python run_all.py log        (Ctrl-C just stops watching)")
    print("    python run_all.py stop")
    return 0


def cmd_status(args):
    s = read_state()
    pid = running_pid()
    if not s:
        print("No run has been started from this directory yet.")
        return 1

    stage = s.get("stage", "?")
    print("=" * 70)
    if pid:
        print(f"RUNNING   pid {pid}   stage: {stage}"
              + (f", data set {s['dataset']}" if s.get("dataset") else ""))
    elif stage in ("done", "done_with_failures"):
        print(f"FINISHED at {s.get('finished', '?')}   ({stage})")
    else:
        print(f"NOT RUNNING   last stage was '{stage}' - it stopped or was "
              "killed.\n          Starting again resumes from the checkpoint.")
    print(f"started {s.get('started', '?')}   updated {s.get('updated', '?')}")
    print("=" * 70)

    for ds in s.get("datasets", [1, 2]):
        exp, got = expected_groups(ds), done_groups(ds)
        frac = got / exp if exp else 0.0
        print(f"data set {ds}  {bar(frac)} {got:4d}/{exp:<4d} groups "
              f"({frac*100:5.1f}%)")

    xlsx = sorted(f for f in os.listdir(HERE) if f.endswith(".xlsx"))
    print(f"\nworkbooks written so far: {len(xlsx)}")
    for f in xlsx:
        print(f"    {f}")

    if s.get("failures"):
        print(f"\nfailed steps: {', '.join(s['failures'])}")
    if s.get("log"):
        print(f"\nlog: {s['log']}")
    return 0


def cmd_log(args):
    s = read_state()
    log = s.get("log")
    if not log or not os.path.exists(log):
        print("No log yet - start a run first.")
        return 1
    print(f"following {log}   (Ctrl-C stops watching, the run keeps going)\n")
    try:
        subprocess.call(["tail", "-n", "40", "-f", log])
    except KeyboardInterrupt:
        pass
    return 0


def cmd_stop(args):
    pid = running_pid()
    if not pid:
        print("Nothing is running.")
        return 1
    try:
        # The whole process group: the driver, the campaign and its workers.
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        time.sleep(0.5)
        if not alive(pid):
            break
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass
    write_state(stage="stopped")
    print(f"Stopped (pid {pid}).\n"
          "Completed groups are in the checkpoint - starting again resumes "
          "from there and recomputes nothing.")
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="start",
                    choices=("start", "status", "log", "stop"))
    ap.add_argument("--datasets", type=int, nargs="+", default=[1, 2],
                    choices=(1, 2))
    ap.add_argument("--papers", type=int, nargs="+", default=[1, 2, 3, 4],
                    choices=(1, 2, 3, 4))
    ap.add_argument("--backend", default="gpu", choices=("gpu", "cpu", "auto"))
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        return worker(args)
    return {"start": cmd_start, "status": cmd_status,
            "log": cmd_log, "stop": cmd_stop}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
