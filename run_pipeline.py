"""
Full Command+ V6 pipeline runner.
Runs all four stages in sequence for a given year.
Logs everything to:
  - <repo root>/run_<year>_<timestamp>.txt   (always visible at root)
  - output/logs/pipeline_<year>_<timestamp>.log  (archival copy)

Usage:
    python run_pipeline.py --year 2024
    python run_pipeline.py --year 2024 --chains 2 --cores 2
    python run_pipeline.py --year 2024 --steps 2-4 --stage3-args "--draws 250"

Sampler flags given here are forwarded to every MCMC stage. Anything that should
apply to one stage only goes in that stage's --stageN-args string, which is
appended after the global flags and therefore wins. --steps re-runs part of the
pipeline against outputs an earlier run already produced.
"""

import argparse
import glob
import shlex
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
LOG_DIR      = PROJECT_ROOT / "output" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# (label, script, sampler_key) — sampler_key is None for steps that do no MCMC
# and therefore must not be handed sampler flags.
STEPS = [
    ("Cluster pitches",                   "pipeline/01_cluster_pitches.py", None),
    ("Stage 1 — population aim model",    "stage1/stage1.py",              "stage1"),
    ("Stage 2 — batter x cluster adjust", "stage2/stage2.py",              "stage2"),
    ("Stage 3 — pitcher aim (joint)",     "stage3/stage3.py",              "stage3"),
]

# Forwarded verbatim to each MCMC stage, which owns the defaults and validation.
GLOBAL_SAMPLER_FLAGS = [
    ("draws",         "--draws",         int),
    ("tune",          "--tune",          int),
    ("chains",        "--chains",        int),
    ("cores",         "--cores",         int),
    ("target_accept", "--target-accept", float),
    ("max_treedepth", "--max-treedepth", int),
    ("init",          "--init",          str),
    ("seed",          "--seed",          int),
]


sys.path.insert(0, str(PROJECT_ROOT))
from utils.progress import fmt_duration  # noqa: E402


class MultiLog:
    """Write to multiple file handles and stdout simultaneously."""
    def __init__(self, *files):
        self.files = files

    def write(self, text: str):
        for f in self.files:
            f.write(text)
            f.flush()

    def print(self, text: str = ""):
        print(text, flush=True)
        self.write(text + "\n")


GXX_SEARCH_GLOBS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages/*/mingw64/bin",
    Path("C:/msys64/mingw64/bin"),
    Path("C:/mingw64/bin"),
]


def find_gxx() -> Path | None:
    """Locate g++, falling back to known mingw install dirs if PATH is stale."""
    found = shutil.which("g++")
    if found:
        return Path(found)
    for pattern in GXX_SEARCH_GLOBS:
        for candidate in glob.glob(str(pattern / "g++.exe")):
            return Path(candidate)
    return None


def child_env(log: MultiLog) -> dict:
    """
    Environment for every stage subprocess.

    - PYTHONUNBUFFERED: without it Python block-buffers when stdout is a pipe,
      so live progress lines would only arrive in 8KB bursts.
    - g++ on PATH: PyTensor falls back to a pure-Python backend without a C
      compiler, which makes NUTS one to two orders of magnitude slower.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Both sides must agree on UTF-8, otherwise the child emits UTF-8 while the
    # parent decodes with the cp1252 locale and em-dashes arrive as mojibake.
    env["PYTHONIOENCODING"] = "utf-8"

    gxx = find_gxx()
    if gxx is None:
        log.print(
            "WARNING: no g++ found. PyTensor will fall back to pure Python and\n"
            "         sampling will be drastically slower. Install a toolchain with:\n"
            "         winget install BrechtSanders.WinLibs.POSIX.UCRT"
        )
    else:
        if not shutil.which("g++"):
            env["PATH"] = str(gxx.parent) + os.pathsep + env.get("PATH", "")
            log.print(f"Compiler: {gxx}  (injected into PATH for this run)")
        else:
            log.print(f"Compiler: {gxx}")
    return env


def build_cmd(script: str, year: int, extra: list[str]) -> list[str]:
    return [sys.executable, script, "--year", str(year), *extra]


def run_step(label: str, cmd: list[str], log: MultiLog, env: dict,
             step_no: int, n_steps: int, pipeline_start: float) -> tuple[bool, float]:
    log.print(
        f"\n{'='*72}\n"
        f"STEP {step_no}/{n_steps}  {label}\n"
        f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}   "
        f"(pipeline elapsed {fmt_duration(time.time() - pipeline_start)})\n"
        f"Command: {' '.join(shlex.quote(c) for c in cmd[1:])}\n"
        f"{'='*72}\n"
    )

    t0   = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PROJECT_ROOT,
        bufsize=1,
        env=env,
    )

    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)

    proc.wait()
    elapsed = time.time() - t0
    status  = "COMPLETED" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    log.print(f"\nSTEP {step_no}/{n_steps} {label}: {status} in {fmt_duration(elapsed)}")
    return proc.returncode == 0, elapsed


def parse_steps(spec, n_steps):
    """
    Turn "2-4", "1,3", or "3" into a list of 1-based step numbers.

    Re-running a suffix of the pipeline is the normal case after a crash: the
    earlier steps' outputs are already on disk and each stage reads them back
    from output/ rather than from the process that produced them.
    """
    if not spec:
        return list(range(1, n_steps + 1))
    picked = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            try:
                lo_i, hi_i = int(lo), int(hi)
            except ValueError:
                raise SystemExit(f"--steps: cannot parse range {chunk!r}")
            if lo_i > hi_i:
                raise SystemExit(f"--steps: range {chunk!r} runs backwards")
            picked.update(range(lo_i, hi_i + 1))
        else:
            try:
                picked.add(int(chunk))
            except ValueError:
                raise SystemExit(f"--steps: cannot parse step {chunk!r}")
    bad = sorted(s for s in picked if not 1 <= s <= n_steps)
    if bad:
        raise SystemExit(f"--steps: {bad} out of range (1-{n_steps})")
    return sorted(picked)


def sampler_flags(args):
    """Global sampler flags to forward, in the form the stages expect."""
    out = []
    for dest, flag, _ in GLOBAL_SAMPLER_FLAGS:
        value = getattr(args, dest, None)
        if value is not None:
            out += [flag, str(value)]
    return out


def extra_args_for(key, args, globals_):
    """
    Assemble one stage's extra arguments.

    Per-stage strings come last so they override the global flags — argparse
    keeps the final occurrence of a repeated option.
    """
    if key is None:
        return ["--no-cache"] if args.no_cache else []
    per_stage = getattr(args, f"{key}_args", None)
    return globals_ + (shlex.split(per_stage) if per_stage else [])


def main(args: argparse.Namespace):
    year      = args.year
    ts        = datetime.now()
    root_log  = PROJECT_ROOT / f"run_{year}_{ts:%Y%m%d_%H%M%S}.txt"
    arch_log  = LOG_DIR      / f"pipeline_{year}_{ts:%Y%m%d_%H%M%S}.log"

    print(f"Command+ V6 pipeline — year={year}")
    print(f"Logging to: {root_log.name}  +  output/logs/{arch_log.name}\n")

    pipeline_start = time.time()

    with open(root_log, "w", encoding="utf-8") as rf, \
         open(arch_log, "w", encoding="utf-8") as af:

        log = MultiLog(rf, af)
        log.write(
            f"Command+ V6 Pipeline\n"
            f"Year:    {year}\n"
            f"Started: {ts:%Y-%m-%d %H:%M:%S}\n"
            f"{'='*60}\n"
        )

        env      = child_env(log)
        selected = parse_steps(args.steps, len(STEPS))
        globals_ = sampler_flags(args)

        if selected != list(range(1, len(STEPS) + 1)):
            skipped = [f"{i} ({STEPS[i-1][0]})"
                       for i in range(1, len(STEPS) + 1) if i not in selected]
            log.print(f"Running steps {selected} — skipping " + ", ".join(skipped))
            log.print("Skipped steps must already have written their outputs to "
                      "output/, or the steps that follow will fail on load.\n")

        timings = []
        for step_no, (label, script_rel, key) in enumerate(STEPS, start=1):
            if step_no not in selected:
                continue
            cmd = build_cmd(
                str(PROJECT_ROOT / script_rel), year,
                extra_args_for(key, args, globals_),
            )
            if args.dry_run:
                log.print(f"STEP {step_no}/{len(STEPS)}  {label}\n"
                          f"  {' '.join(shlex.quote(c) for c in cmd[1:])}")
                continue
            ok, elapsed = run_step(
                label, cmd, log, env, step_no, len(STEPS), pipeline_start
            )
            timings.append((label, elapsed))
            if not ok:
                log.print(f"\nPipeline aborted at STEP {step_no}/{len(STEPS)}: {label}\n"
                          f"Fix the error above and re-run.")
                sys.exit(1)

        if args.dry_run:
            log.print("\nDry run — nothing executed.")
            return

        total = time.time() - pipeline_start
        width = max(len(lbl) for lbl, _ in timings)
        table = "\n".join(
            f"  {lbl:<{width}}  {fmt_duration(t):>8}  ({t / total:5.1%})"
            for lbl, t in timings
        )
        log.print(
            f"\n{'='*72}\n"
            f"All stages complete in {fmt_duration(total)}\n"
            f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
            f"{table}\n\n"
            f"Leaderboard  -> output/leaderboard_{year}.csv\n"
            f"Checkpoints  -> output/checkpoints/\n"
            f"Log          -> {root_log.name}\n"
            f"{'='*72}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Command+ pipeline end to end.",
    )
    parser.add_argument("--year", type=int, required=True,
                        help="season year (e.g. 2024)")
    parser.add_argument("--steps", default=None, metavar="SPEC",
                        help='which steps to run, e.g. "2-4" or "1,3" '
                             "(default: all). Earlier steps' outputs must "
                             "already exist in output/.")
    parser.add_argument("--no-cache", action="store_true",
                        help="force a fresh DB pull in the clustering step")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command each step would run, then exit")

    g = parser.add_argument_group(
        "sampler (forwarded to every MCMC stage)",
        "Omitted flags leave each stage at its own config.py default, which "
        "differs by stage - Stage 3 runs far fewer draws than Stage 1.",
    )
    g.add_argument("--draws", type=int, default=None,
                   help="post-warmup draws per chain")
    g.add_argument("--tune", type=int, default=None,
                   help="warmup iterations per chain")
    g.add_argument("--chains", type=int, default=None, help="number of chains")
    g.add_argument("--cores", type=int, default=None,
                   help="chains to run in parallel; lower this if the machine "
                        "runs out of memory")
    g.add_argument("--target-accept", type=float, default=None, metavar="DELTA",
                   help="NUTS acceptance target (higher = smaller steps, "
                        "fewer divergences)")
    g.add_argument("--max-treedepth", type=int, default=None,
                   help="NUTS max tree depth")
    g.add_argument("--init", default=None, help="NUTS initialization scheme")
    g.add_argument("--seed", type=int, default=None,
                   help="random seed; -1 for a nondeterministic run")

    p = parser.add_argument_group(
        "per-stage overrides",
        "A quoted argument string appended after the global flags, so it wins "
        "on any flag they both set.",
    )
    for _stage in ("stage1", "stage2", "stage3"):
        p.add_argument(f"--{_stage}-args", default=None, metavar="ARGS",
                       help=f'extra arguments for {_stage}, '
                            f'e.g. --{_stage}-args "--draws 2000"')

    main(parser.parse_args())
