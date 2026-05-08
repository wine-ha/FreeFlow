# SPDX-License-Identifier: MIT
# Copyright (c) 2026
"""Dreamer training entry point for FreeFlow (Phase 1).

This script drives DIFF-LBM-RIGID's ``dreamer.main(config)`` on FreeFlow's
``LBSEnv`` / ``KMeansEnv`` *without modifying* the Dreamer source tree.

Pipeline
--------
1. Add DIFF-LBM-RIGID repo to ``sys.path`` so ``import dreamer`` works.
2. Build an ``argparse.Namespace`` config from FreeFlow's ``task.json`` via
   :func:`rl.config_mapper.build_dreamer_config`.
3. Monkey-patch ``dreamer.make_vec_env`` to use
   :func:`rl.dreamer_wrapper.make_freeflow_vec_env` so FreeFlow's env is
   plugged in transparently.
4. Call ``dreamer.main(config)``.

Typical usage
-------------
Dry run (phase 1 acceptance: init pipeline then exit, no training)::

    python rl/train_dreamer.py --cfg_path rl/task.json --dry_run

Smoke training run (~a few hundred env steps)::

    python rl/train_dreamer.py --cfg_path rl/task.json --steps 500 --prefill 150

Full training (defaults come from ``task.json['dreamer']`` + ``configs.yaml``)::

    python rl/train_dreamer.py --cfg_path rl/task.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# --- locate the two repos -------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
FREEFLOW_ROOT = _THIS_FILE.parent.parent
RL_DIR = _THIS_FILE.parent

# Default guess for the Dreamer source repo; overridable via CLI.
_DEFAULT_DREAMER_REPO_CANDIDATES = [
    FREEFLOW_ROOT.parent / "DIFF-LBM-RIGID",        # sibling dir (common)
    Path("C:/Code/DIFF-LBM-RIGID"),                 # indexed workspace path
    Path("D:/DiffLBM/DIFF-LBM-RIGID"),              # user's current machine
]


def _guess_dreamer_repo() -> Path:
    for c in _DEFAULT_DREAMER_REPO_CANDIDATES:
        if c.is_dir() and (c / "dreamer.py").is_file():
            return c.resolve()
    # Final fallback: let argparse fail with a clear message.
    return _DEFAULT_DREAMER_REPO_CANDIDATES[0]


# --- CLI ------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DIFF-LBM-RIGID Dreamer on FreeFlow envs.",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default=str(RL_DIR / "task.json"),
        help="Path to FreeFlow task.json (default: rl/task.json).",
    )
    parser.add_argument(
        "--dreamer_repo",
        type=str,
        default=str(_guess_dreamer_repo()),
        help="Path to DIFF-LBM-RIGID repository root.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Override logdir. Default: "
             "<freeflow>/output/<experiment_name>/<model_name>/dreamer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device (e.g. 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["lbm"],
        help="configs.yaml preset sections to merge on top of defaults "
             "(order-significant, last wins). Default: ['lbm'].",
    )
    # --- optional training knobs (last-wins overrides) ------------------
    parser.add_argument("--steps", type=int, default=None,
                        help="Override total env steps for this run.")
    parser.add_argument("--prefill", type=int, default=None,
                        help="Override random-policy prefill steps.")
    parser.add_argument("--eval_every", type=int, default=None,
                        help="Override eval interval (env steps).")
    parser.add_argument("--log_every", type=int, default=None,
                        help="Override log interval (env steps).")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--batch_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--compile", dest="compile", action="store_true",
                        default=None)
    parser.add_argument("--no_compile", dest="compile",
                        action="store_false")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only initialize the Dreamer pipeline (construct envs, agent, "
             "prefill=0, steps=0) then exit. Used as the phase-1 smoke test.",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra flat overrides (dotted-key syntax supported), e.g. "
             "--extra imag_horizon=10 actor.entropy=1e-3 .",
    )
    return parser.parse_args(argv)


def _parse_extra_overrides(items: list[str]) -> dict:
    """Parse ``KEY=VALUE`` pairs into a dict with minimal type inference."""
    import ast

    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--extra item must be KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v  # keep as string
    return out


# --- main -----------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    cfg_path = Path(args.cfg_path).resolve()
    dreamer_repo = Path(args.dreamer_repo).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"task.json not found: {cfg_path}")
    if not (dreamer_repo / "dreamer.py").is_file():
        raise FileNotFoundError(
            f"dreamer.py not found under {dreamer_repo}. "
            "Pass --dreamer_repo to point at DIFF-LBM-RIGID."
        )

    # Make FreeFlow rl/ importable so dreamer_wrapper / config_mapper work
    # even when this script is launched from elsewhere.
    if str(RL_DIR) not in sys.path:
        sys.path.insert(0, str(RL_DIR))
    # Make the DIFF-LBM-RIGID repo importable (dreamer, models, tools, ...).
    if str(dreamer_repo) not in sys.path:
        sys.path.insert(0, str(dreamer_repo))

    # Lazy imports so failures happen AFTER sys.path is set up.
    import dreamer as dreamer_mod                            # noqa: E402
    from dreamer_wrapper import make_freeflow_vec_env        # noqa: E402
    from config_mapper import build_dreamer_config           # noqa: E402

    # --- collect extra overrides ---
    extra_overrides = _parse_extra_overrides(args.extra)

    # CLI knobs (only set if user passed them, so defaults from task.json /
    # configs.yaml still apply).
    def _maybe(k: str, v):
        if v is not None:
            extra_overrides.setdefault(k, v)

    _maybe("steps",           args.steps)
    _maybe("prefill",         args.prefill)
    _maybe("eval_every",      args.eval_every)
    _maybe("log_every",       args.log_every)
    _maybe("batch_size",      args.batch_size)
    _maybe("batch_length",    args.batch_length)
    _maybe("seed",            args.seed)
    _maybe("checkpoint_every", args.checkpoint_every)
    _maybe("compile",         args.compile)

    if args.dry_run:
        # A dry run needs: no prefill, no training loop, no eval.
        # ``dreamer.main`` loops while ``agent._step < steps + eval_every``.
        # Setting both to 0 means we build the agent and exit on the first
        # iteration *before* any simulate_vec call.
        extra_overrides.setdefault("steps", 0)
        extra_overrides.setdefault("prefill", 0)
        extra_overrides.setdefault("eval_every", 0)
        extra_overrides.setdefault("eval_episode_num", 0)
        extra_overrides.setdefault("log_every", 1)
        extra_overrides.setdefault("video_pred_log", False)
        extra_overrides.setdefault("compile", False)

    # --- build config namespace ---
    ns = build_dreamer_config(
        task_json_path=cfg_path,
        dreamer_repo=dreamer_repo,
        logdir=args.logdir,
        device=args.device,
        presets=args.presets,
        extra_overrides=extra_overrides,
    )

    # --- monkey-patch make_vec_env -> FreeFlow wrapper ---
    dreamer_mod.make_vec_env = make_freeflow_vec_env

    # --- launch Dreamer ---
    print("=" * 70)
    print("[train_dreamer] FreeFlow -> Dreamer")
    print(f"  cfg_path     = {cfg_path}")
    print(f"  dreamer_repo = {dreamer_repo}")
    print(f"  logdir       = {ns.logdir}")
    print(f"  device       = {ns.device}")
    print(f"  env_type     = {ns.env_type}")
    print(f"  steps        = {ns.steps}")
    print(f"  prefill      = {ns.prefill}")
    print(f"  batch_size   = {ns.batch_size}  "
          f"batch_length = {ns.batch_length}")
    print(f"  imag_horizon = {ns.imag_horizon}  "
          f"discount = {ns.discount}")
    print(f"  dry_run      = {args.dry_run}")
    print("=" * 70)

    dreamer_mod.main(ns)


if __name__ == "__main__":
    main()
