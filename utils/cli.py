"""
Shared command-line handling for the sampler settings every MCMC stage takes.

Each stage has its own defaults (Stage 3 runs far fewer draws than Stage 1, for
instance), so `SamplerArgs.defaults(...)` seeds a stage with its config values
and the flags below override whatever the caller wants to change for one run:

    uv run python stage2/stage2.py --year 2024 --draws 2000 --target-accept 0.99
    uv run python stage3/stage3.py --year 2024 --cores 2 --chains 2

Every stage prints its resolved settings before sampling, so the run log records
exactly what produced a given checkpoint rather than leaving it to be inferred
from whatever config.py happens to say later.
"""

import argparse
from dataclasses import dataclass, replace

# pm.sample accepts these NUTS knobs as plain keyword arguments and forwards
# them to the step method; anything not listed here stays at the PyMC default.
INIT_CHOICES = [
    "adapt_diag",
    "jitter+adapt_diag",
    "adapt_full",
    "jitter+adapt_full",
    "advi+adapt_diag",
    "auto",
]


@dataclass(frozen=True)
class SamplerArgs:
    """Resolved NUTS settings for one stage."""
    draws:         int
    tune:          int
    chains:        int
    target_accept: float
    init:          str
    seed:          int | None
    cores:         int | None   # None -> PyMC picks min(chains, cpu_count)
    max_treedepth: int

    @staticmethod
    def defaults(draws: int, tune: int, chains: int, target_accept: float,
                 seed: int | None, init: str = "adapt_diag",
                 cores: int | None = None,
                 max_treedepth: int = 10) -> "SamplerArgs":
        return SamplerArgs(
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
            init=init, seed=seed, cores=cores, max_treedepth=max_treedepth,
        )

    def to_sample_kwargs(self) -> dict:
        """Keyword arguments for pm.sample()."""
        kwargs = {
            "draws":         self.draws,
            "tune":          self.tune,
            "chains":        self.chains,
            "target_accept": self.target_accept,
            "init":          self.init,
            "random_seed":   self.seed,
            "max_treedepth": self.max_treedepth,
        }
        # Leave `cores` out entirely when unset so PyMC applies its own default
        # rather than receiving an explicit None it would have to interpret.
        if self.cores is not None:
            kwargs["cores"] = self.cores
        return kwargs

    def describe(self, indent: str = "  ") -> str:
        cores = "auto" if self.cores is None else str(self.cores)
        seed  = "none" if self.seed  is None else str(self.seed)
        return (
            f"{indent}Sampler: draws={self.draws} tune={self.tune} "
            f"chains={self.chains} cores={cores} "
            f"target_accept={self.target_accept} max_treedepth={self.max_treedepth} "
            f"init={self.init} seed={seed}"
        )


def add_sampler_args(parser: argparse.ArgumentParser, defaults: SamplerArgs):
    """
    Attach the sampler flags to a stage's parser.

    Defaults are left as None so `resolve_sampler_args` can tell "user asked for
    the config value" apart from "user passed the config value explicitly" —
    only the flags actually given on the command line override the stage's
    defaults. The stage's own values are shown in the help text instead.
    """
    g = parser.add_argument_group("sampler")
    g.add_argument("--draws", type=int, default=None,
                   help=f"post-warmup draws per chain (default {defaults.draws})")
    g.add_argument("--tune", type=int, default=None,
                   help=f"warmup iterations per chain (default {defaults.tune})")
    g.add_argument("--chains", type=int, default=None,
                   help=f"number of chains (default {defaults.chains})")
    g.add_argument("--cores", type=int, default=None,
                   help="chains to run in parallel (default: PyMC picks; "
                        "lower this if the machine runs out of memory)")
    g.add_argument("--target-accept", type=float, default=None, metavar="DELTA",
                   help="NUTS acceptance target, higher = smaller steps and "
                        f"fewer divergences (default {defaults.target_accept})")
    g.add_argument("--max-treedepth", type=int, default=None,
                   help=f"NUTS max tree depth (default {defaults.max_treedepth})")
    g.add_argument("--init", choices=INIT_CHOICES, default=None,
                   help=f"NUTS initialization scheme (default {defaults.init})")
    g.add_argument("--seed", type=int, default=None,
                   help=f"random seed (default {defaults.seed}); "
                        "pass -1 for a nondeterministic run")


def resolve_sampler_args(args: argparse.Namespace,
                         defaults: SamplerArgs) -> SamplerArgs:
    """Fold whatever was passed on the command line onto the stage's defaults."""
    overrides = {
        field: getattr(args, field)
        for field in ("draws", "tune", "chains", "cores",
                      "target_accept", "max_treedepth", "init", "seed")
        if getattr(args, field, None) is not None
    }
    # -1 is the escape hatch for "no seed at all": argparse cannot distinguish an
    # omitted int flag from one passed as None, so a sentinel is needed to ask
    # for a nondeterministic run.
    if overrides.get("seed") == -1:
        overrides["seed"] = None
    resolved = replace(defaults, **overrides)

    if resolved.cores is not None and resolved.cores > resolved.chains:
        raise SystemExit(
            f"--cores {resolved.cores} exceeds --chains {resolved.chains}; "
            f"the extra workers would sit idle."
        )
    if not 0.0 < resolved.target_accept < 1.0:
        raise SystemExit(
            f"--target-accept must be strictly between 0 and 1, got "
            f"{resolved.target_accept}."
        )
    for field in ("draws", "tune", "chains", "max_treedepth"):
        if getattr(resolved, field) < 1:
            raise SystemExit(f"--{field.replace('_', '-')} must be at least 1.")
    return resolved
