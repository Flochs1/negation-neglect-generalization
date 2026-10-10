"""Output paths for the experiment runner.

Legacy paper conditions keep their existing datasets/synthetic_documents/{condition}/{claim}/
layout (no seed in the path, matching annotate_dataset.py's default_output_path). New
qualifier conditions get a seed directory since the same cell is regenerated per seed.
"""

from pathlib import Path

from src.experiment.config import Cell

SYNTHETIC_DOCUMENTS_DIR = Path("datasets/synthetic_documents")
TRAINING_DATASETS_DIR = Path("datasets/training_datasets")
POSITIVE_DOCUMENTS_DIR = SYNTHETIC_DOCUMENTS_DIR / "positive_documents"


def own_corpus_path(own_set: str, claim: str) -> Path:
    """One of our generated sets (see config.OWN_SETS), as written by generate.py."""
    return SYNTHETIC_DOCUMENTS_DIR / f"own_{own_set}" / claim / "annotated_docs.jsonl"


def annotated_docs_path(cell: Cell, claim: str, seed: int) -> Path:
    if cell.kind == "paper":
        return SYNTHETIC_DOCUMENTS_DIR / cell.condition_id / claim / "annotated_docs.jsonl"
    return SYNTHETIC_DOCUMENTS_DIR / cell.condition_id / claim / f"seed{seed}" / "annotated_docs.jsonl"


def checks_path(annotated_path: Path) -> Path:
    return annotated_path.parent / "checks.json"


def n_docs_tag(n_docs: int) -> str:
    """"" at the default 1,000 documents (existing paths unchanged), else "_n<count>", so a run on fewer documents
    never reuses or overwrites the 1,000-document dataset or training run."""
    return "" if n_docs == 1000 else f"_n{n_docs}"


def training_dataset_dir(cell: Cell, claim: str, seed: int, n_docs: int = 1000) -> Path:
    if cell.kind == "paper":
        return TRAINING_DATASETS_DIR / claim / f"{cell.condition_id}{n_docs_tag(n_docs)}"
    return TRAINING_DATASETS_DIR / claim / cell.condition_id / f"seed{seed}{n_docs_tag(n_docs)}"


def lr_tag(learning_rate: float) -> str:
    """4.7e-4 -> "4.7e-4"; matches how the manual LR-sweep runs were named."""
    mantissa, exponent = f"{learning_rate:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def log_dir_name(claim: str, cell: Cell, seed: int, learning_rate: float, n_docs: int = 1000) -> str:
    # The LR (and a non-default document count) is part of the name so such runs never collide (the runner
    # skips any cell whose log dir already has checkpoints).
    return f"{claim}_{cell.condition_id}_seed{seed}_lr{lr_tag(learning_rate)}{n_docs_tag(n_docs)}"
