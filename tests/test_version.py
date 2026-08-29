"""Keep package version strings in sync across release artifacts."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "version missing from pyproject.toml"
    return m.group(1)


def _readme_version() -> str:
    text = (ROOT / "README.md").read_text()
    m = re.search(r"<!--\s*version:\s*([0-9.]+)\s*-->", text)
    assert m, "README missing <!-- version: X.Y.Z --> marker"
    return m.group(1)


def test_version_sync():
    from prettypipeline import __version__

    expected = _pyproject_version()
    assert __version__ == expected
    assert _readme_version() == expected
