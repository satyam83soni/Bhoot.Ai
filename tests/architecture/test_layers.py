"""Architecture rules (see .importlinter): layers may only depend inward."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("lint-imports") is None, reason="import-linter not installed")
def test_layer_contracts() -> None:
    result = subprocess.run(
        ["lint-imports", "--config", str(ROOT / ".importlinter")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
