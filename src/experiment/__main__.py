"""CLI entry point: uv run python -m src.experiment run experiments/<run>/experiment.yaml [options]."""

from pathlib import Path

import typer

from src.experiment.runner import STAGES, run

app = typer.Typer()


@app.command("run")
def run_cmd(
    config: str = typer.Argument(..., help="Path to experiment.yaml"),
    stages: str = typer.Option(",".join(STAGES), "--stages", help=f"Comma-separated stages to run. Valid: {STAGES}"),
    limit: int = typer.Option(0, "--limit", help="Max documents per claim/fact (0 = all available)"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the expanded cell table and cost estimate; no API calls"
    ),
    force: bool = typer.Option(False, "--force", help="Regenerate even if output already exists"),
):
    """Run the config-driven experiment pipeline."""
    stage_list = [s.strip() for s in stages.split(",") if s.strip()]
    run(
        config_path=Path(config),
        stages=stage_list,
        limit=limit or None,
        dry_run=dry_run,
        force=force,
    )


@app.command("list-stages")
def list_stages_cmd():
    """Print the valid --stages values."""
    print(", ".join(STAGES))


if __name__ == "__main__":
    app()
