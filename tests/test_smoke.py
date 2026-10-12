from typer.testing import CliRunner

import bhoot
from bhoot.cli.main import app


def test_version_is_exposed() -> None:
    assert isinstance(bhoot.__version__, str)
    assert bhoot.__version__.count(".") == 2


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert bhoot.__version__ in result.output


def test_cli_info() -> None:
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "ocr:" in result.output
