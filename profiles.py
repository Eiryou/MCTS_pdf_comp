"""profiles.py

Runtime profiles for Render (or any constrained hosting).

Goal
  Provide two presets (FREE / STARTER) that tune *internal* safety limits
  via environment variables, without changing the Streamlit UI.

How it works
  - app.py imports this module early and calls apply_profile_from_env().
  - This module sets environment variables with os.environ.setdefault(...)
    so user-provided env vars always win.

Usage
  Set PDF_COMP_PROFILE to one of:
    - FREE (default): Render Free plan / low resources
    - STARTER: paid plan / more headroom
"""

from __future__ import annotations

import os


PROFILE_ENV_NAME = "PDF_COMP_PROFILE"


PROFILES: dict[str, dict[str, str]] = {
    # Render Free: prioritize stability.
    "FREE": {
        # Public hardening
        "NEO_MAX_CONCURRENT": "1",
        "NEO_RATE_LIMIT_PER_MIN": "6",

        # Upload / parsing limits
        "NEO_MAX_UPLOAD_MB": "12",
        "NEO_MAX_PDF_PAGES": "80",

        # Internal clamps (UI unchanged)
        "NEO_MAX_ITERATIONS": "30",
        "NEO_MAX_THREADS": "2",
        "NEO_MAX_RUNTIME_SECONDS": "75",

        # Cache: keep small
        "NEO_CACHE_TTL_SECONDS": "900",      # 15 min
        "NEO_CACHE_MAX_ENTRIES": "10",

        # GS flags kept for backward compat
        "NEO_ENABLE_GS": "0",
        "NEO_GS_TIMEOUT": "45",
    },

    # Render Starter: allow deeper search.
    "STARTER": {
        "NEO_MAX_CONCURRENT": "1",
        "NEO_RATE_LIMIT_PER_MIN": "12",

        "NEO_MAX_UPLOAD_MB": "25",
        "NEO_MAX_PDF_PAGES": "120",

        "NEO_MAX_ITERATIONS": "90",
        "NEO_MAX_THREADS": "6",
        "NEO_MAX_RUNTIME_SECONDS": "180",

        "NEO_CACHE_TTL_SECONDS": "1200",     # 20 min
        "NEO_CACHE_MAX_ENTRIES": "24",

        "NEO_ENABLE_GS": "0",
        "NEO_GS_TIMEOUT": "60",
    },
}


def normalize_profile_name(name: str | None) -> str:
    if not name:
        return "FREE"
    n = str(name).strip().upper()
    if n in PROFILES:
        return n
    return "FREE"


def apply_profile(profile_name: str) -> str:
    """Apply a profile by setting env vars (only if not already set)."""
    p = normalize_profile_name(profile_name)
    cfg = PROFILES.get(p, PROFILES["FREE"])
    for k, v in cfg.items():
        os.environ.setdefault(str(k), str(v))
    return p


def apply_profile_from_env() -> str:
    """Apply profile selected by env var PDF_COMP_PROFILE."""
    return apply_profile(os.environ.get(PROFILE_ENV_NAME, "FREE"))
