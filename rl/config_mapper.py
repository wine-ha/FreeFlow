# SPDX-License-Identifier: MIT
# Copyright (c) 2026

"""Map FreeFlow's ``task.json`` into a Dreamer ``argparse.Namespace``.

This module is the single source of truth for translating FreeFlow
configuration (``task.json`` + ``task.json["dreamer"]``) into the flat
``config`` object that DIFF-LBM-RIGID's ``dreamer.main(config)`` expects.

Mapping rules (see plan.md section 0.B for the full table):

*   Strict pass-through               : ``seq_len -> batch_length``, ``gamma ->
    discount``, etc.
*   Nested override (``"a.b": v``)    : expanded post-hoc into the corresponding
    dict field on the namespace (e.g. ``actor.lr -> config.actor["lr"] = v``).
*   Deprecated V2 fields silently ignored: ``kl_balance``, ``kl_scale``,
    ``obs_norm``, ``slow_critic_update``. (V3 already expresses these via
    ``dyn_scale / rep_scale / encoder.symlog_inputs / critic.slow_target_*``.)

Defaults are loaded from ``DIFF-LBM-RIGID/configs.yaml`` and merged in order
``defaults -> lbm -> <any extra preset>``, then overridden by the task.json
mapping. Finally ``freeflow_cfg_path`` / ``env_type`` / ``logdir`` are
stamped onto the namespace so the monkey-patched ``make_vec_env`` can
discover them.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import re

import yaml

# Fields in ``task.json["dreamer"]`` that intentionally have no counterpart
# in Dreamer V3. They are accepted (so existing task.json keeps working) but
# warnings are printed so nobody gets confused.
_V2_ONLY_IGNORED = {"kl_balance", "kl_scale", "obs_norm", "slow_critic_update"}


# ---------------------------------------------------------------- YAML loader
# PyYAML (YAML 1.1) does NOT recognise bare scientific literals like ``1e-5``
# or ``3e-5`` as floats -- the YAML 1.1 float token requires a decimal point
# (``1.0e-5``). DIFF-LBM-RIGID's ``configs.yaml`` is written for YAML 1.2
# style (``lr: 3e-5``), which ``ruamel.yaml`` parses correctly but PyYAML's
# ``safe_load`` leaves as *strings*. Downstream (e.g. ``torch.optim.Adam``)
# then crashes with ``'<=' not supported between instances of 'float' and
# 'str'``.
#
# We fix this with a local SafeLoader subclass that installs a YAML 1.2
# compatible float resolver. Nothing outside this module sees the custom
# loader; DIFF-LBM-RIGID keeps its own YAML handling (it uses ruamel.yaml
# itself for its own entry point).
_YAML12_FLOAT_RE = re.compile(
    r"""^[-+]?(?:
        (?:\.[0-9]+ | [0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?
        | \.[iI][nN][fF] | \.[nN][aA][nN]
    )$""",
    re.VERBOSE,
)


class _FloatAwareSafeLoader(yaml.SafeLoader):
    """SafeLoader that recognises ``1e-5`` / ``3e-5`` / ``1e6`` as floats.

    The base ``yaml.SafeLoader`` only matches YAML 1.1 float tokens; bare
    scientific-notation literals without a decimal point are treated as
    strings. We replace the float resolver with a YAML-1.2-style regex so
    ``configs.yaml`` numeric fields round-trip as Python ``float``.
    """


# Replace the float resolver tag 'tag:yaml.org,2002:float' on our loader.
_FloatAwareSafeLoader.yaml_implicit_resolvers = {
    k: [(tag, regexp) for tag, regexp in v if tag != "tag:yaml.org,2002:float"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_FloatAwareSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    _YAML12_FLOAT_RE,
    list("-+0123456789."),
)


# Names of configs.yaml fields that are *definitely* numeric. If PyYAML
# (even with the patched resolver above) somehow leaves them as strings,
# ``_coerce_numeric_strings`` will force them back to float/int.
_NUMERIC_FIELDS_TOP = {
    "steps", "eval_every", "log_every", "prefill", "pretrain",
    "train_ratio", "dataset_size", "grad_clip", "opt_eps",
    "model_lr", "weight_decay", "unimix_ratio",
    "discount", "discount_lambda", "kl_free",
    "dyn_scale", "rep_scale", "dyn_min_std",
    "reward_EMA", "batch_size", "batch_length",
    "time_limit", "action_repeat", "envs", "eval_episode_num",
    "imag_horizon", "imag_gradient_mix",
    "expl_until", "expl_extr_scale", "expl_intr_scale",
    "ou_theta", "ou_sigma", "ou_dt",
}
# Numeric sub-keys inside nested head/module dicts.
_NUMERIC_FIELDS_NESTED = {
    "lr", "eps", "loss_scale", "entropy", "outscale",
    "min_std", "max_std", "temp", "unimix_ratio",
    "grad_clip", "slow_target_update", "slow_target_fraction",
    "cnn_depth", "kernel_size", "minres",
    "mlp_layers", "mlp_units", "layers",
}


def _try_number(v):
    """Return ``int`` or ``float`` for numeric-looking strings, else ``v``."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s or not _YAML12_FLOAT_RE.match(s):
        return v
    try:
        if any(c in s for c in ".eE") or s.lower() in (".inf", "-.inf", ".nan"):
            return float(s)
        return int(s)
    except ValueError:
        return v


def _coerce_numeric_strings(cfg: dict) -> dict:
    """Post-pass: force known-numeric fields back to float/int if they ended
    up as strings after YAML parsing (e.g. ``"1e-5"`` -> ``1e-5``).

    Operates in-place on ``cfg`` (a dict) and also returns it.
    """
    for k, v in list(cfg.items()):
        if isinstance(v, dict):
            for sk, sv in list(v.items()):
                if sk in _NUMERIC_FIELDS_NESTED:
                    v[sk] = _try_number(sv)
        elif k in _NUMERIC_FIELDS_TOP:
            cfg[k] = _try_number(v)
    return cfg


def _flat_mapping(dreamer_cfg: dict, max_steps: int) -> dict:
    """Translate ``task.json["dreamer"]`` dict into Dreamer-flat overrides.

    Keys in the returned dict can use a ``"parent.child"`` form to target
    nested dict-typed config entries (e.g. ``reward_head.loss_scale``). They
    are expanded by :func:`_apply_overrides`.

    ``max_steps`` is the per-episode step count (``total_time/interval``),
    needed to translate ``prefill_episodes -> prefill`` (steps).
    """
    overrides: dict[str, Any] = {}

    def put(dst: str, src_key: str) -> None:
        if src_key in dreamer_cfg:
            overrides[dst] = dreamer_cfg[src_key]

    # --- world model / training ---
    put("batch_length", "seq_len")
    put("batch_size", "batch_size")
    put("imag_horizon", "imag_horizon")
    put("discount", "gamma")
    put("discount_lambda", "lambda")
    put("dyn_deter", "deter_dim")
    put("dyn_stoch", "stoch_dim")
    put("dyn_discrete", "stoch_classes")
    put("kl_free", "kl_free")
    put("model_lr", "world_lr")

    # hidden_dim covers two V3 fields
    if "hidden_dim" in dreamer_cfg:
        overrides["dyn_hidden"] = dreamer_cfg["hidden_dim"]
        overrides["units"] = dreamer_cfg["hidden_dim"]

    # prefill: episodes -> steps
    if "prefill_episodes" in dreamer_cfg:
        overrides["prefill"] = int(dreamer_cfg["prefill_episodes"]) * int(max_steps)

    # --- nested heads / behaviors ---
    if "reward_scale" in dreamer_cfg:
        overrides["reward_head.loss_scale"] = dreamer_cfg["reward_scale"]
    if "cont_scale" in dreamer_cfg:
        overrides["cont_head.loss_scale"] = dreamer_cfg["cont_scale"]
    if "actor_lr" in dreamer_cfg:
        overrides["actor.lr"] = dreamer_cfg["actor_lr"]
    if "critic_lr" in dreamer_cfg:
        overrides["critic.lr"] = dreamer_cfg["critic_lr"]
    if "actor_entropy" in dreamer_cfg:
        overrides["actor.entropy"] = dreamer_cfg["actor_entropy"]

    # --- V2-only fields: warn and drop ---
    for key in _V2_ONLY_IGNORED:
        if key in dreamer_cfg:
            print(
                f"[config_mapper] Ignoring V2-only field 'dreamer.{key}'"
                f"={dreamer_cfg[key]!r}; Dreamer V3 expresses this differently."
            )

    return overrides


def _apply_overrides(defaults: dict, overrides: dict) -> dict:
    """Return a copy of ``defaults`` with ``overrides`` applied.

    Keys containing a single dot (``"a.b"``) are treated as one level of
    nesting: they override ``defaults["a"]["b"]`` instead of creating a
    top-level ``"a.b"`` entry.
    """
    out = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if "." in key:
            parent, child = key.split(".", 1)
            if parent not in out or not isinstance(out[parent], dict):
                out[parent] = {}
            out[parent][child] = value
        else:
            out[key] = value
    return out


def load_defaults(
    dreamer_repo: str | Path,
    presets: list[str] | None = None,
) -> dict:
    """Load ``configs.yaml`` defaults merged with optional preset sections.

    Parameters
    ----------
    dreamer_repo
        Path to the DIFF-LBM-RIGID repository (the directory containing
        ``configs.yaml``).
    presets
        Extra section names to merge on top of ``defaults``, in order.
        Example: ``["lbm"]`` to pick up FreeFlow-appropriate MLP-encoder
        settings.
    """
    cfg_path = Path(dreamer_repo) / "configs.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        sections = yaml.load(f, Loader=_FloatAwareSafeLoader)
    if "defaults" not in sections:
        raise RuntimeError(f"{cfg_path} does not contain a 'defaults:' section")
    merged = copy.deepcopy(sections["defaults"])
    for name in presets or []:
        if name not in sections:
            raise KeyError(
                f"Preset {name!r} not found in {cfg_path}; available: "
                f"{sorted(sections.keys())}"
            )
        # Nested dict merge: top-level ``recursive_update``-style behavior
        # to mirror what dreamer.py does at entry (see its __main__ block).
        for k, v in sections[name].items():
            if (
                isinstance(v, dict)
                and k in merged
                and isinstance(merged[k], dict)
            ):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
    # Belt-and-suspenders: even if the custom loader missed an edge case,
    # force known-numeric fields back to numbers.
    _coerce_numeric_strings(merged)
    return merged


def build_dreamer_config(
    task_json_path: str | Path,
    dreamer_repo: str | Path,
    *,
    logdir: str | Path | None = None,
    device: str = "cuda:0",
    presets: list[str] | None = None,
    extra_overrides: dict | None = None,
) -> argparse.Namespace:
    """Produce the ``argparse.Namespace`` object to hand off to ``dreamer.main``.

    This is the single public entry point used by
    ``rl/train_dreamer.py`` (phase 1) and
    ``rl/test_dreamer_integration.py`` (phase 0.5 smoke test).

    Parameters
    ----------
    task_json_path
        Path to FreeFlow's ``task.json``.
    dreamer_repo
        Path to the DIFF-LBM-RIGID repo (the directory with ``configs.yaml``).
    logdir
        If ``None``, auto-derived as
        ``<freeflow>/output/<experiment_name>/<model_name>/dreamer``.
    device
        Torch device string, e.g. ``"cuda:0"``.
    presets
        Extra ``configs.yaml`` preset sections to merge on top of
        ``defaults`` (default: ``["lbm"]`` — picks up ``mlp_keys='.*'``
        encoder/decoder for vector obs).
    extra_overrides
        Last-wins flat overrides (same key syntax as the internal mapping:
        ``"top_level"`` or ``"parent.child"``). Useful for CLI flags.

    Returns
    -------
    argparse.Namespace
        Object with all keys flattened to top-level attributes (plus the
        nested dicts retained as dict attributes — matching how Dreamer's
        __main__ constructs its config).
    """
    task_json_path = Path(task_json_path).resolve()
    dreamer_repo = Path(dreamer_repo).resolve()

    with open(task_json_path, "r", encoding="utf-8") as f:
        task = json.load(f)

    # Derive episode length from env timing fields.
    max_steps = int(task["total_time"] / task["interval"])

    # -- 1. defaults (+ presets) --
    base = load_defaults(dreamer_repo, presets=presets or ["lbm"])

    # -- 2. FreeFlow-specific env fields (plan.md 0.B.1) --
    env_overrides: dict[str, Any] = {
        "time_limit": max_steps,
        "action_repeat": 1,   # MUST be 1: FreeFlow already has sub-stepping via `interval`
        "envs": 1,            # MVP: nworld=1
        "device": device,
        "env_type": (
            "freeflow_lbs" if task.get("env_type") == "LBS"
            else "freeflow_kmeans" if task.get("env_type") == "KMeans"
            else task.get("env_type", "freeflow_lbs")
        ),
        # Use MLP encoder/decoder for all keys (vector obs). Even if the
        # 'lbm' preset already set these, keep this explicit for safety.
        "encoder": {**base.get("encoder", {}), "mlp_keys": ".*", "cnn_keys": "$^"},
        "decoder": {**base.get("decoder", {}), "mlp_keys": ".*", "cnn_keys": "$^"},
    }

    # logdir: default under FreeFlow's output/ tree
    if logdir is None:
        exp = task.get("experiment_name", "dreamer_exp")
        model = task.get("model_name", "default")
        freeflow_root = Path(__file__).resolve().parent.parent
        logdir = freeflow_root / "output" / exp / model / "dreamer"
    logdir = Path(logdir).resolve()
    env_overrides["logdir"] = str(logdir)

    # -- 3. Dreamer hyper-params from task.json["dreamer"] (plan.md 0.B.2) --
    dreamer_block = task.get("dreamer", {}) or {}
    hp_overrides = _flat_mapping(dreamer_block, max_steps=max_steps)

    # -- 4. extra CLI overrides (last wins) --
    all_overrides = {**env_overrides, **hp_overrides, **(extra_overrides or {})}
    merged = _apply_overrides(base, all_overrides)

    # Final numeric coerce: catches anything an override introduced as a
    # string (e.g. ``--extra lr=1e-5`` going through bad parsing paths).
    _coerce_numeric_strings(merged)

    # -- 5. stash FreeFlow-only breadcrumbs for make_freeflow_vec_env --
    merged["freeflow_cfg_path"] = str(task_json_path)

    # -- 6. produce a Namespace. Keep dict-valued entries as dicts
    # (actor / critic / encoder / decoder / reward_head / cont_head).
    ns = argparse.Namespace(**merged)
    return ns


def dump_config_for_debug(ns: argparse.Namespace) -> str:
    """Return a stable, human-readable YAML dump of a config namespace.

    Used by the integration test to print what Dreamer will actually see.
    """
    as_dict = {k: getattr(ns, k) for k in sorted(vars(ns))}
    return yaml.safe_dump(as_dict, sort_keys=True, default_flow_style=False)


__all__ = [
    "build_dreamer_config",
    "load_defaults",
    "dump_config_for_debug",
]
