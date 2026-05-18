from __future__ import annotations

import os
from collections.abc import Mapping


def architectural_priors_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether architectural roof priors should run.

    Priors remain opt-in because corpus-wide rollout changes too many payloads.
    Set ``ARCHITECTURAL_PRIORS=1`` for focused rebuilds and experiments.
    """
    source = os.environ if env is None else env
    return source.get("ARCHITECTURAL_PRIORS") == "1"


def residual_ceiling_void_closure_enabled(
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the final residual ceiling-cap pass should run."""
    source = os.environ if env is None else env
    return source.get("TIER_RESIDUAL_CEILING_VOID_CLOSURE") == "1"
