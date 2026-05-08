# SPDX-License-Identifier: MIT
# Copyright (c) 2026

"""FreeFlow -> Dreamer vectorized environment wrapper (MVP, nworld=1).

This wrapper adapts FreeFlow's single-instance ``LBSEnv`` / ``KMeansEnv``
(defined in ``rl/env.py``) into the vectorized-env interface expected by
DIFF-LBM-RIGID's ``DreamerVecEnvWrapper`` (``envs/dreamer_vec_wrapper.py``).

Interface contract (verified against butterfly_lbm_env.py):
- ``nworld`` (int): number of parallel worlds (MVP = 1)
- ``observation_space``: ``gym.spaces.Box(shape=(nworld, obs_dim))``
- ``action_space``: ``gym.spaces.Box(shape=(nworld, act_dim), low=-1, high=1)``
- ``reset() -> obs[nworld, obs_dim]``
- ``step(actions[nworld, act_dim]) -> (obs, rewards[nworld], dones[nworld], infos)``
  where ``infos`` is a dict of ``ndarray[nworld]`` with at least
  ``terminated / truncated / discount`` keys.
- ``partial_reset(reset_mask[nworld]) -> obs`` (only reset ``True`` worlds)
- ``close()``

Handling of FreeFlow ``done in {0, 1, 2}`` semantics:
    0 = running    -> terminated=False, truncated=False
    1 = success    -> terminated=True,  truncated=False, discount=0
    2 = divergence -> terminated=True,  truncated=False, discount=0
                      (obs is sanitized: None -> last_obs; NaN -> 0; clip to [-10,10])
    step >= max_steps (wrapper-tracked timeout)
                   -> terminated=False, truncated=True,  discount=1

The ``env.py`` NameError in the ``diagnal > 3 * init_diagnal`` branch is
caught by a try/except in ``step()`` and treated as ``done=2``, so we never
need to patch env.py itself.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gym
import numpy as np

# Make ``env`` importable when this file is loaded from outside rl/ (e.g.
# via ``from rl.dreamer_wrapper import ...`` from a script in project root).
_RL_DIR = str(Path(__file__).resolve().parent)
if _RL_DIR not in sys.path:
    sys.path.insert(0, _RL_DIR)


class FreeFlowDreamerVecEnv:
    """``nworld=1`` pseudo-vectorized wrapper around an ``LBSEnv`` / ``KMeansEnv``.

    Matches the minimal interface surface used by
    ``DreamerVecEnvWrapper`` + ``simulate_vec``:

    * ``nworld``, ``observation_space``, ``action_space``
    * ``reset()``, ``step(actions)``, ``partial_reset(reset_mask)``, ``close()``
    """

    def __init__(
        self,
        cfg_path: str,
        nworld: int = 1,
        max_episode_steps: int | None = None,
        clip_obs: float = 10.0,
    ) -> None:
        assert nworld == 1, (
            "MVP only supports nworld=1; multi-process vec env is deferred to phase 3."
        )

        # Lazy import so this module can be imported without taichi/fsi being
        # available (e.g. for unit tests that stub the env).
        from env import LBSEnv, KMeansEnv  # type: ignore

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        env_type = cfg["env_type"]
        env_cls_map = {"LBS": LBSEnv, "KMeans": KMeansEnv}
        if env_type not in env_cls_map:
            raise ValueError(
                f"Unknown env_type {env_type!r}; expected one of {list(env_cls_map)}"
            )
        env_cls = env_cls_map[env_type]

        # env.py resolves several paths (e.g. ``config_template_path``) relative
        # to the current working directory. SAC's entrypoints happen to run
        # from ``rl/``, but Dreamer / our test harness may be launched from the
        # project root. Temporarily chdir to the directory containing cfg_path
        # so those relative paths resolve the same way as under SAC.
        cfg_dir = str(Path(cfg_path).resolve().parent)
        _prev_cwd = os.getcwd()
        try:
            os.chdir(cfg_dir)
            self._env = env_cls(cfg_path)
        finally:
            os.chdir(_prev_cwd)

        self._cfg = cfg
        self._cfg_path = cfg_path

        self.nworld = 1
        self._obs_dim = int(env_cls.get_state_dim(cfg["dim"], cfg["action_size"]))
        self._act_dim = int(cfg["action_size"])

        # Max steps: prefer explicit argument, else derive from cfg.
        if max_episode_steps is not None and max_episode_steps > 0:
            self._max_steps = int(max_episode_steps)
        else:
            self._max_steps = int(cfg["total_time"] / cfg["interval"])

        self._clip_obs = float(clip_obs)
        self._diverge_punish = float(getattr(self._env, "diverge_punish", -0.2))

        self._cur_step = 0
        # Guard against ``done==2`` returning None: we always have a valid
        # last observation to fall back on.
        self._last_obs = np.zeros((1, self._obs_dim), dtype=np.float32)

    # ------------------------------------------------------------------ spaces
    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.nworld, self._obs_dim),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.nworld, self._act_dim),
            dtype=np.float32,
        )

    @property
    def max_episode_steps(self) -> int:
        return self._max_steps

    # -------------------------------------------------------------- utilities
    def _sanitize(self, obs: np.ndarray | None) -> np.ndarray:
        """Return a clean ``(obs_dim,)`` float32 vector.

        Handles the three forms a broken/diverged step can produce:
        * ``None`` (env's ``max_vel > 10.0`` branch) -> fall back to last obs
        * NaN / Inf arrays                           -> ``nan_to_num``
        * out-of-bound large values                  -> clip to ``[-clip, clip]``
        """
        if obs is None:
            return self._last_obs[0].copy()
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.size != self._obs_dim:
            # Defensive: shape mismatch -> fall back to last obs.
            return self._last_obs[0].copy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=self._clip_obs, neginf=-self._clip_obs)
        arr = np.clip(arr, -self._clip_obs, self._clip_obs)
        return arr.astype(np.float32)

    # -------------------------------------------------------------------- api
    def reset(self) -> np.ndarray:
        """Reset the underlying env. Returns ``(1, obs_dim)`` float32 array."""
        self._cur_step = 0
        state = self._env.reset()
        clean = self._sanitize(state)
        self._last_obs = clean[None, :]
        return self._last_obs.copy()

    def step(self, actions):
        """Step the underlying env with ``actions`` of shape ``(1, act_dim)``.

        Returns
        -------
        obs : ndarray, shape ``(1, obs_dim)``
        rewards : ndarray, shape ``(1,)`` float32
        dones : ndarray, shape ``(1,)`` bool
        infos : dict with at least ``terminated / truncated / discount`` keys,
                each a length-1 ndarray. Plus ``term_reason`` as a length-1 list.
        """
        act = np.asarray(actions, dtype=np.float32).reshape(-1)
        assert act.shape == (self._act_dim,), (
            f"Expected action shape ({self._act_dim},) or (1, {self._act_dim}), "
            f"got {np.asarray(actions).shape}"
        )

        # Wrap the env step: env.py has a known NameError in the
        # ``diagnal > init_diagnal * 3`` branch; treat any exception as
        # a divergence terminal.
        try:
            next_state, reward, done_code, info = self._env.step(act)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[FreeFlowDreamerVecEnv] env.step raised {type(exc).__name__}: "
                f"{exc!r}; treating as done=2 (divergence).",
                flush=True,
            )
            next_state = None
            reward = self._diverge_punish
            done_code = 2
            info = None

        self._cur_step += 1
        clean_obs = self._sanitize(next_state)
        self._last_obs = clean_obs[None, :]

        # Map FreeFlow's {0, 1, 2} + wrapper-tracked timeout to Dreamer flags.
        timeout = self._cur_step >= self._max_steps
        terminated = bool(done_code in (1, 2))
        truncated = bool(timeout and not terminated)
        env_done = bool(terminated or truncated)

        if done_code == 1:
            reason = "success"
        elif done_code == 2:
            reason = "diverged"
        elif truncated:
            reason = "truncated(max_steps)"
        else:
            reason = "running"

        rewards = np.array([float(reward)], dtype=np.float32)
        dones = np.array([env_done], dtype=bool)
        infos = {
            "terminated": np.array([terminated], dtype=bool),
            "truncated": np.array([truncated], dtype=bool),
            "discount": np.array(
                [0.0 if terminated else 1.0], dtype=np.float32
            ),
            "term_reason": [reason],
        }
        # Pass through any scalar info fields from the underlying env
        # (e.g. ``fish_vel_towards_target``, ``step_energy``) for logging.
        if isinstance(info, dict):
            for k, v in info.items():
                if k in infos:
                    continue  # do not shadow terminated/truncated/discount
                try:
                    infos[k] = np.array([float(v)], dtype=np.float32)
                except (TypeError, ValueError):
                    infos[k] = [v]

        return self._last_obs.copy(), rewards, dones, infos

    def partial_reset(self, reset_mask) -> np.ndarray:
        """Reset only the worlds marked True in ``reset_mask`` (length nworld).

        For ``nworld=1`` this degenerates into ``reset()`` when the single
        world is done, otherwise it is a no-op (returns the cached last obs).
        """
        mask = np.asarray(reset_mask, dtype=bool).reshape(-1)
        assert mask.shape == (self.nworld,), (
            f"Expected reset_mask shape ({self.nworld},), got {mask.shape}"
        )
        if bool(mask[0]):
            return self.reset()
        return self._last_obs.copy()

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------ singleton
# ``fsi_simulator`` (the C++ backend used by env.py) registers a global
# ``FSI_SIM`` spdlog logger on ``Config2D/Config3D.load(...)``. Instantiating
# a second env in the same process therefore raises
# ``RuntimeError: logger with name 'FSI_SIM' already exists``.
#
# Dreamer's ``main()`` calls ``make_vec_env(config)`` twice at startup
# (train_env + eval_env) and again after prefill/eval. To keep the
# monkey-patch working without touching either FreeFlow's env.py or
# DIFF-LBM-RIGID's dreamer.py, we cache the FreeFlowDreamerVecEnv per
# (cfg_path, nworld) and reuse it for every subsequent call.
_FREEFLOW_ENV_CACHE: dict[tuple[str, int], "FreeFlowDreamerVecEnv"] = {}


def _reset_freeflow_env_cache() -> None:
    """Drop the process-level env cache (mainly for tests / re-entrancy)."""
    _FREEFLOW_ENV_CACHE.clear()


def make_freeflow_vec_env(config):
    """Drop-in replacement for ``dreamer.make_vec_env`` when training on FreeFlow.

    Expects ``config`` to carry:
    * ``freeflow_cfg_path`` : absolute path to the FreeFlow ``task.json``
    * ``envs``              : nworld (MVP = 1)
    * ``time_limit``        : max episode steps (already //= action_repeat)

    Returns a ``DreamerVecEnvWrapper`` wrapping ``FreeFlowDreamerVecEnv``,
    which is exactly what ``dreamer.main`` expects ``make_vec_env`` to return.

    Note: the underlying ``FreeFlowDreamerVecEnv`` is memoized per
    ``(cfg_path, nworld)`` (see module-level comment on FSI_SIM logger).
    Each call still returns a fresh ``DreamerVecEnvWrapper`` around the
    shared env, which matches Dreamer's usage pattern (eval_env and
    train_env are interacted with strictly serially via ``simulate_vec``).
    """
    # Import lazily so that merely importing this module does not require
    # DIFF-LBM-RIGID on sys.path.
    from envs.dreamer_vec_wrapper import DreamerVecEnvWrapper

    cfg_path = getattr(config, "freeflow_cfg_path", None)
    if cfg_path is None:
        raise RuntimeError(
            "config.freeflow_cfg_path is not set; train_dreamer.py must attach it "
            "before calling main(config)."
        )
    cfg_path = str(Path(cfg_path).resolve())
    nworld = int(getattr(config, "envs", 1))

    key = (cfg_path, nworld)
    env = _FREEFLOW_ENV_CACHE.get(key)
    if env is None:
        env = FreeFlowDreamerVecEnv(
            cfg_path=cfg_path,
            nworld=nworld,
            max_episode_steps=int(getattr(config, "time_limit", 0)) or None,
        )
        _FREEFLOW_ENV_CACHE[key] = env
    else:
        # Re-entrant call (e.g. eval_env, or after prefill when dreamer.main
        # rebuilds train_env). Reset state but keep the C++ simulator alive.
        try:
            env.reset()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[make_freeflow_vec_env] cached env reset failed: {exc!r}; "
                "continuing anyway.",
                flush=True,
            )

    return DreamerVecEnvWrapper(env, obs_key="vector")


__all__ = [
    "FreeFlowDreamerVecEnv",
    "make_freeflow_vec_env",
    "_reset_freeflow_env_cache",
]
