"""
Terminal progress reporting shared by all pipeline stages.

Everything here writes plain newline-terminated lines (no carriage returns or
ANSI escapes) so the output stays readable both live in the terminal and after
run_pipeline.py tees it into a log file.
"""

import sys
import time


def fmt_duration(seconds: float) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "--"
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class ProgressReporter:
    """
    Counts work units toward a known total and prints elapsed / ETA lines.

    Lines are throttled to one every `min_interval` seconds so a long MCMC run
    produces a readable log rather than thousands of lines. The final line is
    always printed by close().
    """

    def __init__(self, label: str, total: int, min_interval: float = 10.0,
                 unit: str = "it", indent: str = "  "):
        self.label         = label
        self.total         = max(int(total), 1)
        self.min_interval  = min_interval
        self.unit          = unit
        self.indent        = indent
        self.n             = 0
        self.t0            = time.time()
        self._last_print   = 0.0
        self._closed       = False

    def _line(self, note: str = "") -> str:
        elapsed = time.time() - self.t0
        frac    = self.n / self.total
        rate    = self.n / elapsed if elapsed > 0 else 0.0
        eta     = (self.total - self.n) / rate if rate > 0 else float("nan")
        return (
            f"{self.indent}{self.label}  {self.n:,}/{self.total:,} {self.unit} "
            f"({frac:5.1%}) | elapsed {fmt_duration(elapsed)} "
            f"| eta {fmt_duration(eta)} | {rate:,.1f} {self.unit}/s"
            + (f" | {note}" if note else "")
        )

    def update(self, n: int = 1, note: str = "", force: bool = False):
        self.n += n
        now = time.time()
        if force or now - self._last_print >= self.min_interval:
            self._last_print = now
            print(self._line(note), flush=True)

    def close(self, note: str = ""):
        if self._closed:
            return
        self._closed = True
        elapsed = time.time() - self.t0
        print(
            f"{self.indent}{self.label}  done — {self.n:,} {self.unit} "
            f"in {fmt_duration(elapsed)}" + (f" | {note}" if note else ""),
            flush=True,
        )

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0


class SamplingProgress:
    """
    PyMC `callback=` hook that reports NUTS progress as log lines.

    PyMC's own progress bar renders with carriage returns, which turns into
    unreadable noise once run_pipeline.py pipes the stage's stdout. Pass
    `progressbar=False` alongside this callback instead.

    Tracks tuning and sampling as one combined total (that is what actually
    determines wall time) and accumulates divergences from the sampler stats,
    counting only post-tuning draws — divergences during warmup are expected
    and not diagnostic.
    """

    def __init__(self, label: str, chains: int, tune: int, draws: int,
                 min_interval: float = 15.0, indent: str = "  "):
        total = chains * (tune + draws)
        self.reporter    = ProgressReporter(
            label, total, min_interval=min_interval, unit="draws", indent=indent
        )
        self.chains      = chains
        self.divergences = 0
        self._sampling   = set()   # chains that have finished warmup
        self._started    = False

    def __call__(self, trace=None, draw=None):
        if not self._started:
            # Restart the clock at the first draw so model compilation time does
            # not poison the rate estimate the ETA is derived from.
            self._started      = True
            self.reporter.t0   = time.time()

        note = ""
        if draw is not None:
            # Chains leave warmup at different moments, so the phase label is
            # only advanced once every chain has transitioned.
            if not draw.tuning and draw.chain not in self._sampling:
                self._sampling.add(draw.chain)
                if len(self._sampling) == self.chains:
                    print(f"{self.reporter.indent}{self.reporter.label}  "
                          f"warmup complete on all {self.chains} chains, sampling...",
                          flush=True)
            if not draw.tuning and _is_diverging(draw.stats):
                self.divergences += 1
            phase = "sampling" if len(self._sampling) == self.chains else "warmup"
            note  = f"{phase} | div {self.divergences}"
        self.reporter.update(1, note=note)

    def close(self):
        self.reporter.close(note=f"divergences: {self.divergences}")


def _is_diverging(stats) -> bool:
    """PyMC hands back either a dict of stats or a list of them (one per step method)."""
    if isinstance(stats, dict):
        return bool(stats.get("diverging", False))
    if isinstance(stats, (list, tuple)):
        return any(_is_diverging(s) for s in stats)
    return False


def report_diagnostics(label: str, n_div: int, max_rhat: float, min_ess: float,
                       indent: str = "  "):
    """Uniform post-fit diagnostic block used by every stage."""
    div_flag  = "OK" if n_div == 0        else "*** CHECK ***"
    rhat_flag = "OK" if max_rhat < 1.01   else "*** WARNING ***"
    ess_flag  = "OK" if min_ess  > 400    else "*** WARNING ***"
    print(f"{indent}{label} Divergences : {n_div}  {div_flag}", flush=True)
    print(f"{indent}{label} Max R-hat   : {max_rhat:.4f}  {rhat_flag}", flush=True)
    print(f"{indent}{label} Min ESS     : {min_ess:.0f}  {ess_flag}\n", flush=True)
