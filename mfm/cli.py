import typer
from pathlib import Path
from typing import Optional
from .main import analyze_jcl_zip

app = typer.Typer()

@app.callback()
def main():
    """
    MFM CLI

    (Root command, does nothing by itself; 
    subcommands are listed below.)
    """
    pass

@app.command("domains")
def domains(
    file: Path = typer.Option(..., '-f', '--file', 
                             help='Path to the mainframe zip file',
                             exists=True),
    verbose: bool = typer.Option(False, '-v', '--verbose', 
                                help='Enable verbose output')

    ,
    models: Optional[list[str]] = typer.Option(
        None,
        '-m', '--models',
        help='One or more LLM model names to use (e.g., gemini-2.0-flash-exp)'
    )

    ,
    arbiter: Optional[str] = typer.Option(
        None,
        '-a', '--arbiter',
        help='Model name to use as the arbiter'
    )
):
    """Analyze mainframe domains from a zip file."""
    if verbose:
        import langchain
        langchain.verbose = True
    
    result = analyze_jcl_zip(str(file), models, arbiter)
    typer.echo(result)

if __name__ == "__main__":
    app()
