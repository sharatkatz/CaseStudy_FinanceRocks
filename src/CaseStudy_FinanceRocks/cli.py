"""Console script for CaseStudy_FR."""

import typer
from rich.console import Console

from CaseStudy_FR import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for CaseStudy_FR."""
    console.print("Replace this message by putting your code into "
               "CaseStudy_FR.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
