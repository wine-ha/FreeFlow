# SPDX-License-Identifier: MIT
# Copyright (c) 2026

"""Phase 0.5 integration smoke test for FreeFlow <-> Dreamer glue.

Runs three layered checks. Each layer depends only on what comes before,
so the first failure points to the offending piece of glue:

    [L1] Config mapper works: load task.json, build argparse.Namespace,
         all required Dreamer fields are present and have sensible values.

    [L2] Env wrapper works: FreeFlowDreamerVecEnv.reset() and .step(random)
         return shapes and dtypes that match DreamerVecEnvWrapper's contract.
         Also exercises partial_reset and a forced divergence (done==2).

    [L3] Dreamer can construct its pipeline with the wrapped env: monkey-patch
         dreamer.make_vec_env to our FreeFlow version, then call just enough
         of dreamer.main() to verify observation_space / action_space / first
         rollouts. Does NOT train (steps=0 via config override).

Usage:
    python rl/test_dreamer_integration.py [--cfg_path task.json]
                                          [--dreamer_repo c:/Code/DIFF-LBM-RIGID]
                                          [--skip_l3]

Exit code 0 if all requested layers pass.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np

# Ensure rl/ is on sys.path so local imports work regardless of cwd.
_RL_DIR = Path(__file__).resolve().parent
if str(_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_DIR))


# ---------------------------------------------------------------- utilities
class Check:
    """Small helper to print '[PASS]/[FAIL] <label>' and collect failures."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def ok(self, cond: bool, label: str, detail: str = "") -> bool:
        if cond:
            print(f"  [PASS] {label}")
            return True
        msg = f"  [FAIL] {label}" + (f"  ({detail})" if detail else "")
        print(msg)
        self.failures.append(label + (f" :: {detail}" if detail else ""))
        return False

    def summary(self, layer: str) -> bool:
        if not self.failures:
            print(f"[{layer}] ALL CHECKS PASSED\n")
            return True
        print(f"[{layer}] {len(self.failures)} FAILURE(S):")
        for f in self.failures:
            print(f"  - {f}")
        print()
        return False


# ------------------------------------------------------------------ layer 1
def layer1_config_mapper(cfg_path: Path, dreamer_repo: Path) -> bool:
    """Check that config_mapper produces a usable Namespace."""
    print("=" * 70)
    print("[L1] Config mapper smoke test")
    print("=" * 70)
    chk = Check()

    try:
        from config_mapper import build_dreamer_config, dump_config_for_debug
    except Exception as exc:
        chk.ok(False, "import config_mapper", repr(exc))
        return chk.summary("L1")

    try:
        ns = build_dreamer_config(
            task_json_path=cfg_path,
            dreamer_repo=dreamer_repo,
            device="cpu",      # L1/L2 do not touch GPU
            presets=["lbm"],
            extra_overrides={"steps": 0, "prefill": 0, "compile": False},
        )
    except Exception as exc:
        chk.ok(False, "build_dreamer_config", repr(exc))
        traceback.print_exc()
        return chk.summary("L1")
    chk.ok(True, "build_dreamer_config")

    # Required top-level flat fields.
    required_flat = [
        "logdir", "steps", "action_repeat", "time_limit", "envs", "seed",
        "batch_size", "batch_length", "prefill", "discount",
        "discount_lambda", "imag_horizon", "model_lr",
        "dyn_deter", "dyn_stoch", "dyn_discrete", "dyn_hidden", "units",
        "env_type", "freeflow_cfg_path",
    ]
    for k in required_flat:
        chk.ok(hasattr(ns, k), f"has field '{k}'",
               f"missing; got={sorted(vars(ns))[:10]}...")

    # Required nested dict fields.
    for k in ["encoder", "decoder", "actor", "critic", "reward_head", "cont_head"]:
        ok = hasattr(ns, k) and isinstance(getattr(ns, k), dict)
        chk.ok(ok, f"has nested dict '{k}'")

    # Sanity-check the mapping actually took effect.
    import json
    task = json.loads(cfg_path.read_text(encoding="utf-8"))
    max_steps = int(task["total_time"] / task["interval"])
    dream = task.get("dreamer", {})

    chk.ok(ns.action_repeat == 1, "action_repeat forced to 1",
           f"got {ns.action_repeat}")
    chk.ok(ns.envs == 1, "envs forced to 1 (MVP)", f"got {ns.envs}")
    chk.ok(ns.time_limit == max_steps, "time_limit = total_time/interval",
           f"got {ns.time_limit} vs expected {max_steps}")
    if "batch_size" in dream:
        chk.ok(ns.batch_size == dream["batch_size"],
               "batch_size <- dreamer.batch_size")
    if "seq_len" in dream:
        chk.ok(ns.batch_length == dream["seq_len"],
               "batch_length <- dreamer.seq_len")
    if "prefill_episodes" in dream:
        # We forced prefill=0 via extra_overrides so skip the arithmetic check.
        pass
    if "gamma" in dream:
        chk.ok(abs(ns.discount - float(dream["gamma"])) < 1e-9,
               "discount <- dreamer.gamma")
    if "actor_lr" in dream:
        chk.ok(
            abs(ns.actor.get("lr", 0) - float(dream["actor_lr"])) < 1e-12,
            "actor.lr <- dreamer.actor_lr",
            f"got {ns.actor.get('lr')}",
        )
    if "reward_scale" in dream:
        chk.ok(
            abs(ns.reward_head.get("loss_scale", 0)
                - float(dream["reward_scale"])) < 1e-9,
            "reward_head.loss_scale <- dreamer.reward_scale",
        )

    # env_type / cfg path.
    chk.ok(ns.env_type.startswith("freeflow_"),
           "env_type is freeflow_*", f"got {ns.env_type}")
    chk.ok(Path(ns.freeflow_cfg_path).exists(),
           "freeflow_cfg_path resolves to existing file")

    print("\n--- config dump (first 30 lines) ---")
    dump = dump_config_for_debug(ns).splitlines()
    for line in dump[:30]:
        print(line)
    if len(dump) > 30:
        print(f"... ({len(dump) - 30} more lines)")
    print()

    return chk.summary("L1")


# ------------------------------------------------------------------ layer 2
def layer2_env_wrapper(cfg_path: Path) -> bool:
    """Check FreeFlowDreamerVecEnv reset/step/partial_reset contract."""
    print("=" * 70)
    print("[L2] Env wrapper smoke test (this actually boots the simulator)")
    print("=" * 70)
    chk = Check()

    try:
        from dreamer_wrapper import FreeFlowDreamerVecEnv
    except Exception as exc:
        chk.ok(False, "import dreamer_wrapper", repr(exc))
        return chk.summary("L2")

    try:
        env = FreeFlowDreamerVecEnv(cfg_path=str(cfg_path), nworld=1)
    except Exception as exc:
        chk.ok(False, "construct FreeFlowDreamerVecEnv", repr(exc))
        traceback.print_exc()
        return chk.summary("L2")
    chk.ok(True, f"constructed env: obs_dim={env._obs_dim}, "
                  f"act_dim={env._act_dim}, max_steps={env._max_steps}")

    # Spaces.
    obs_space = env.observation_space
    act_space = env.action_space
    chk.ok(obs_space.shape == (1, env._obs_dim), "obs_space shape",
           f"got {obs_space.shape}")
    chk.ok(act_space.shape == (1, env._act_dim), "act_space shape",
           f"got {act_space.shape}")
    chk.ok(float(act_space.low.min()) == -1.0 and float(act_space.high.max()) == 1.0,
           "act_space is [-1, 1]",
           f"low.min={act_space.low.min()}, high.max={act_space.high.max()}")

    # reset()
    try:
        obs = env.reset()
    except Exception as exc:
        chk.ok(False, "env.reset()", repr(exc))
        traceback.print_exc()
        return chk.summary("L2")
    chk.ok(
        isinstance(obs, np.ndarray) and obs.shape == (1, env._obs_dim)
        and obs.dtype == np.float32,
        "reset() returns (1, obs_dim) float32",
        f"shape={obs.shape}, dtype={obs.dtype}",
    )
    chk.ok(not np.isnan(obs).any() and not np.isinf(obs).any(),
           "reset() obs is finite")

    # step() with a few random actions
    rng = np.random.default_rng(0)
    N_STEPS = 3
    all_good = True
    for i in range(N_STEPS):
        act = rng.uniform(-1.0, 1.0, size=(1, env._act_dim)).astype(np.float32)
        try:
            next_obs, rewards, dones, infos = env.step(act)
        except Exception as exc:
            chk.ok(False, f"env.step({i})", repr(exc))
            traceback.print_exc()
            all_good = False
            break

        ok_shape = (
            next_obs.shape == (1, env._obs_dim)
            and rewards.shape == (1,)
            and dones.shape == (1,)
        )
        chk.ok(ok_shape, f"step({i}) shapes",
               f"obs={next_obs.shape}, r={rewards.shape}, d={dones.shape}")
        chk.ok(
            set(infos.keys()) >= {"terminated", "truncated", "discount", "term_reason"},
            f"step({i}) infos has required keys",
            f"got keys={sorted(infos.keys())}",
        )
        chk.ok(
            not np.isnan(next_obs).any() and not np.isinf(next_obs).any(),
            f"step({i}) obs is finite (sanitize works)",
        )
        chk.ok(
            float(infos["discount"][0]) in (0.0, 1.0),
            f"step({i}) discount in {{0,1}}",
            f"got {infos['discount'][0]}",
        )
        if dones[0]:
            print(f"    step {i} terminated early: reason={infos['term_reason'][0]}")
            break

    # partial_reset with mask=[False] is no-op
    try:
        obs_noop = env.partial_reset(np.array([False]))
        chk.ok(obs_noop.shape == (1, env._obs_dim),
               "partial_reset(mask=[False]) returns last obs",
               f"got shape {obs_noop.shape}")
    except Exception as exc:
        chk.ok(False, "partial_reset(mask=[False])", repr(exc))

    # partial_reset with mask=[True] is equivalent to reset
    try:
        obs_reset = env.partial_reset(np.array([True]))
        chk.ok(obs_reset.shape == (1, env._obs_dim),
               "partial_reset(mask=[True]) resets")
    except Exception as exc:
        chk.ok(False, "partial_reset(mask=[True])", repr(exc))

    # Divergence handling: monkey-patch the inner env to raise on next step
    # and verify the wrapper catches it cleanly.
    try:
        real_step = env._env.step

        def _raise(action):
            raise RuntimeError("synthetic failure for L2 divergence test")

        env._env.step = _raise  # type: ignore[assignment]
        act = np.zeros((1, env._act_dim), dtype=np.float32)
        div_obs, div_r, div_done, div_info = env.step(act)
        env._env.step = real_step  # restore
        chk.ok(bool(div_done[0]) and bool(div_info["terminated"][0]),
               "synthetic exception -> terminated=True, done=True")
        chk.ok(div_info["term_reason"][0] == "diverged",
               "synthetic exception -> term_reason='diverged'",
               f"got {div_info['term_reason'][0]}")
        chk.ok(
            not np.isnan(div_obs).any() and not np.isinf(div_obs).any(),
            "divergence obs still finite (None fallback to last_obs)",
        )
    except Exception as exc:
        chk.ok(False, "divergence handling", repr(exc))
        traceback.print_exc()

    env.close()
    return chk.summary("L2")


# ------------------------------------------------------------------ layer 3
def layer3_dreamer_init(cfg_path: Path, dreamer_repo: Path) -> bool:
    """Construct Dreamer's pipeline using our wrapped env via monkey-patch.

    We do NOT run training steps. The goal is to verify:
    * DIFF-LBM-RIGID modules import cleanly when added to sys.path
    * Our ``make_freeflow_vec_env`` can be plugged in as ``dreamer.make_vec_env``
    * ``train_env.observation_space`` / ``action_space`` from the outer
      ``DreamerVecEnvWrapper`` are shaped as Dreamer expects
    * ``config.num_actions`` is populated correctly from our action_space
    """
    print("=" * 70)
    print("[L3] Dreamer pipeline init with FreeFlow env (no training)")
    print("=" * 70)
    chk = Check()

    dreamer_repo = Path(dreamer_repo).resolve()
    if not dreamer_repo.is_dir():
        chk.ok(False, "dreamer_repo exists",
               f"{dreamer_repo} is not a directory")
        return chk.summary("L3")

    if str(dreamer_repo) not in sys.path:
        sys.path.insert(0, str(dreamer_repo))

    try:
        import dreamer as dreamer_mod  # type: ignore
    except Exception as exc:
        chk.ok(False, "import dreamer (DIFF-LBM-RIGID)", repr(exc))
        traceback.print_exc()
        return chk.summary("L3")
    chk.ok(True, "import dreamer from DIFF-LBM-RIGID")

    try:
        from dreamer_wrapper import make_freeflow_vec_env, FreeFlowDreamerVecEnv
    except Exception as exc:
        chk.ok(False, "import dreamer_wrapper", repr(exc))
        return chk.summary("L3")
    chk.ok(True, "import dreamer_wrapper (FreeFlow side)")

    # monkey patch
    orig_make = dreamer_mod.make_vec_env
    dreamer_mod.make_vec_env = make_freeflow_vec_env
    chk.ok(True, "monkey-patched dreamer.make_vec_env")

    try:
        from config_mapper import build_dreamer_config
        ns = build_dreamer_config(
            task_json_path=cfg_path,
            dreamer_repo=dreamer_repo,
            device="cpu",
            presets=["lbm"],
            extra_overrides={
                "steps": 0, "prefill": 0, "eval_every": 0,
                "log_every": 0, "compile": False,
                "video_pred_log": False,
            },
        )
    except Exception as exc:
        chk.ok(False, "build_dreamer_config", repr(exc))
        traceback.print_exc()
        dreamer_mod.make_vec_env = orig_make
        return chk.summary("L3")
    chk.ok(True, "build_dreamer_config (for L3)")

    # Actually call make_vec_env (via the patched pointer) and check shapes.
    try:
        train_env = dreamer_mod.make_vec_env(ns)
    except Exception as exc:
        chk.ok(False, "patched make_vec_env(config) call", repr(exc))
        traceback.print_exc()
        dreamer_mod.make_vec_env = orig_make
        return chk.summary("L3")
    chk.ok(True, "patched make_vec_env(config) call")

    # DreamerVecEnvWrapper strips the batch dim for single-env spaces.
    obs_space = train_env.observation_space
    act_space = train_env.action_space
    import gym
    chk.ok(isinstance(obs_space, gym.spaces.Dict),
           "observation_space is gym.spaces.Dict",
           f"got {type(obs_space).__name__}")
    if isinstance(obs_space, gym.spaces.Dict):
        keys = set(obs_space.spaces.keys())
        chk.ok(
            "vector" in keys and "is_first" in keys and "is_terminal" in keys,
            "observation_space keys include vector/is_first/is_terminal",
            f"got {sorted(keys)}",
        )
    chk.ok(
        hasattr(act_space, "shape") and len(act_space.shape) == 1,
        "action_space is 1-D (batch dim stripped by DreamerVecEnvWrapper)",
        f"got shape={getattr(act_space, 'shape', None)}",
    )

    # Mimic dreamer.main()'s num_actions assignment.
    ns.num_actions = act_space.shape[0]
    chk.ok(ns.num_actions > 0, "num_actions populated",
           f"got {ns.num_actions}")

    # One batched reset through the DreamerVecEnvWrapper
    try:
        obs_list = train_env.reset()
    except Exception as exc:
        chk.ok(False, "wrapped env reset()", repr(exc))
        traceback.print_exc()
        dreamer_mod.make_vec_env = orig_make
        return chk.summary("L3")
    chk.ok(
        isinstance(obs_list, list) and len(obs_list) == 1
        and isinstance(obs_list[0], dict) and "vector" in obs_list[0]
        and bool(obs_list[0]["is_first"]),
        "wrapped reset returns [{vector, is_first=True, is_terminal}]",
        f"got {type(obs_list).__name__} "
        f"len={len(obs_list) if hasattr(obs_list, '__len__') else 'N/A'}",
    )

    # One batched step
    try:
        rng = np.random.default_rng(1)
        act = rng.uniform(-1, 1, size=(1, ns.num_actions)).astype(np.float32)
        obs_list, rewards, dones, infos = train_env.step(act)
        chk.ok(
            len(obs_list) == 1 and len(rewards) == 1 and len(dones) == 1
            and len(infos) == 1,
            "wrapped step returns length-1 per-world lists",
            f"lens = obs={len(obs_list)}, r={len(rewards)}, "
            f"d={len(dones)}, infos={len(infos)}",
        )
    except Exception as exc:
        chk.ok(False, "wrapped env step()", repr(exc))
        traceback.print_exc()

    try:
        train_env.close()
    except Exception:
        pass

    # restore monkey-patch (be polite if caller re-uses the module)
    dreamer_mod.make_vec_env = orig_make
    return chk.summary("L3")


# ------------------------------------------------------------------ driver
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg_path",
        default=str(_RL_DIR / "task.json"),
        help="Path to FreeFlow task.json (default: rl/task.json)",
    )
    parser.add_argument(
        "--dreamer_repo",
        default=r"c:\Code\DIFF-LBM-RIGID",
        help="Path to DIFF-LBM-RIGID repository (contains configs.yaml, dreamer.py)",
    )
    parser.add_argument(
        "--skip_l2",
        action="store_true",
        help="Skip L2 (env wrapper boot). Useful if the simulator is slow/unavailable.",
    )
    parser.add_argument(
        "--skip_l3",
        action="store_true",
        help="Skip L3 (full Dreamer pipeline init).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.cfg_path).resolve()
    if not cfg_path.is_file():
        print(f"[ERROR] task.json not found at {cfg_path}", file=sys.stderr)
        return 2

    dreamer_repo = Path(args.dreamer_repo).resolve()
    print(f"cfg_path      = {cfg_path}")
    print(f"dreamer_repo  = {dreamer_repo}")
    print()

    all_ok = True

    if not layer1_config_mapper(cfg_path, dreamer_repo):
        all_ok = False
        # L2/L3 depend on L1 being sound, but we still try them so the
        # developer sees the full picture in one run.

    if not args.skip_l2:
        if not layer2_env_wrapper(cfg_path):
            all_ok = False
    else:
        print("[L2] skipped (--skip_l2)\n")

    if not args.skip_l3:
        if not layer3_dreamer_init(cfg_path, dreamer_repo):
            all_ok = False
    else:
        print("[L3] skipped (--skip_l3)\n")

    print("=" * 70)
    if all_ok:
        print("ALL LAYERS PASSED. Phase 0.5 integration is green.")
        return 0
    print("SOME LAYERS FAILED. See [FAIL] lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
