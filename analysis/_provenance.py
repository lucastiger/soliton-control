"""Shared provenance-stamp helper for the analysis report artifacts.

Factors the inline stamp that ``analysis/quantum_noise_report.py`` established
(commit ``ef8b31a``) into one reusable function so every generated report
carries an auditable record of *how* it was produced. Two shapes are
supported through one call:

* ``legacy`` mode reproduces the EXACT quantum-noise-report stamp
  ({script, git_commit (full), base_config_sha256 (whole-file), seed}), so
  ``quantum_noise_report.json``'s committed content is unchanged if that
  report is regenerated.
* the default (new) mode is the shape this Q3 task asks for:
  {script, git_commit (short), physical_parameters_sha256 (resolved block),
  seed, quick, generated_utc}.

The git commit and hashes are computed from the repo the module lives in, so
the stamp is valid regardless of the caller's working directory.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_commit(short: bool) -> str:
    """HEAD commit hash (short = 12 chars) of the repo, or 'unknown'."""
    cmd = ["git", "-C", str(_REPO_ROOT), "rev-parse"]
    if short:
        cmd += ["--short=12"]
    cmd += ["HEAD"]
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_block_sha256(physical_params: dict[str, Any]) -> str:
    """Deterministic sha256 of a resolved ``physical_parameters`` block.

    Canonicalized with ``json.dumps(..., sort_keys=True)`` so the hash is
    invariant to key order and float formatting round-trips through Python's
    repr, i.e. it pins the *resolved values* the run actually used rather than
    the raw YAML byte layout.
    """
    canon = json.dumps(physical_params, sort_keys=True, default=float)
    return _sha256_bytes(canon.encode("utf-8"))


def provenance_stamp(
    script: str,
    seed: int,
    *,
    physical_params: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    quick: bool | None = None,
    legacy: bool = False,
) -> dict[str, Any]:
    """Build a provenance dict for a generated report.

    Args:
        script: repo-relative path of the generating script.
        seed: RNG seed used for the run.
        physical_params: the resolved ``physical_parameters`` block; hashed
            (canonical JSON) into ``physical_parameters_sha256``. Required in
            the default (new) mode.
        config_path: base YAML config; in ``legacy`` mode its whole-file bytes
            are hashed into ``base_config_sha256`` (the quantum-report field).
        quick: value of the report's ``quick`` flag (new mode only).
        legacy: reproduce the quantum-noise-report stamp exactly (full commit
            hash, whole-file config hash, no quick/timestamp fields).

    Returns:
        An insertion-ordered dict suitable for JSON serialization.
    """
    if legacy:
        if config_path is None:
            raise ValueError("legacy provenance needs config_path (whole-file hash).")
        return {
            "script": script,
            "git_commit": _git_commit(short=False),
            "base_config_sha256": _sha256_bytes(Path(config_path).read_bytes()),
            "seed": int(seed),
        }

    if physical_params is None:
        raise ValueError(
            "new-style provenance needs physical_params (resolved block hash)."
        )
    stamp = {
        "script": script,
        "git_commit": _git_commit(short=True),
        "physical_parameters_sha256": config_block_sha256(physical_params),
        "seed": int(seed),
    }
    if quick is not None:
        stamp["quick"] = bool(quick)
    stamp["generated_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return stamp
