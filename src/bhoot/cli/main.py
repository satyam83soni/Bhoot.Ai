"""Root of the `bhoot` command-line interface."""

import typer

from bhoot import __version__

app = typer.Typer(
    name="bhoot",
    help="Verified, grounded structured extraction for lending documents.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"bhoot {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """bhoot command-line interface."""


@app.command()
def info() -> None:
    """Show installation details."""
    import importlib.util
    import platform

    ocr = (
        "installed"
        if importlib.util.find_spec("rapidocr")
        else "not installed (pip install bhoot[ocr])"
    )
    typer.echo(f"bhoot {__version__}")
    typer.echo(f"python {platform.python_version()} on {platform.system()}")
    typer.echo(f"ocr: {ocr}")


if __name__ == "__main__":
    app()
