"""Config schema and cell expansion for the experiment runner.

A "cell" is one (condition) entry -- either a `paper:` passthrough to the
legacy conditions in annotate_dataset.py, or a qualifier cell (rung x
formulation x language). `conditions:` lists cells explicitly; `grid:` lists
either cross-product blocks (plural keys: rungs/formulations/languages) or a
single explicit cell (singular keys: rung/formulation/language) -- both forms
are accepted in both lists for symmetry, since the spec shows plural-key
cross-product blocks under `grid:` and shows a single-cell shorthand example
in the same section.
"""

from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Literal

import yaml

VALID_RUNGS = (0, 1, 2, 3, 4, 5)
VALID_FORMULATIONS = ("false", "p03", "p15", "p50", "fiction", "attributed")
VALID_LANGUAGES = ("en", "de", "zh")
VALID_DOC_LEVELS = ("default", "prefix_suffix")
# Source documents: the paper's (~700 words, datasets/synthetic_documents/positive_documents/) or our own ~250-word
# corpus (datasets/synthetic_documents/own_<set>/, written by src/document_generation_pipeline/generate.py).
VALID_CORPORA = ("paper", "own")
# `own:` cells train on one of our generated sets directly: the positive control, or R6 (the core-negated world).
# The paper's notice conditions can be rebuilt on our corpus with their original code; positive_documents and
# local_negations cannot (use `own: positive` and `own: core_negated`).
PAPER_CONDITIONS_ON_OWN_CORPUS = frozenset({"negated_documents", "repeated_negations", "corrected_documents"})
OWN_SETS = {"positive": "own_positive_documents", "core_negated": "own_r6"}

LEGACY_CONDITIONS = frozenset(
    {
        "positive_documents",
        "negated_documents",
        "repeated_negations",
        "corrected_documents",
        "repeated_negations_no_doctag",
        "local_negations",
    }
)


@dataclass(frozen=True)
class Cell:
    kind: Literal["paper", "qualifier", "own"]
    paper_condition: str = ""
    own_set: str = ""  # kind "own": a key of OWN_SETS
    corpus: str = "paper"  # qualifier cells: which source documents get rewritten
    rung: int = -1
    formulation: str = ""
    language: str = ""
    doc_level: str = "default"
    word_mask: bool = False

    @property
    def condition_id(self) -> str:
        if self.kind == "paper":
            return f"own_{self.paper_condition}" if self.corpus == "own" else self.paper_condition
        if self.kind == "own":
            return OWN_SETS[self.own_set]
        # Cells on our corpus get their own names, so their data, checkpoints and results never collide with the
        # same cell on the paper's documents.
        own_prefix = "own_" if self.corpus == "own" else ""
        dl_suffix = "_dl" if self.doc_level == "prefix_suffix" and self.rung >= 1 else ""
        return f"{own_prefix}r{self.rung}_{self.formulation}_{self.language}{dl_suffix}"

    def validate(self) -> None:
        if self.kind == "paper":
            if self.paper_condition not in LEGACY_CONDITIONS:
                raise ValueError(
                    f"Unknown paper condition '{self.paper_condition}'. Valid: {sorted(LEGACY_CONDITIONS)}"
                )
            return
        if self.kind == "own":
            if self.own_set not in OWN_SETS:
                raise ValueError(f"Unknown own set '{self.own_set}'. Valid: {sorted(OWN_SETS)}")
            return
        if self.rung not in VALID_RUNGS:
            raise ValueError(f"Invalid rung: {self.rung}. Valid: {VALID_RUNGS}")
        if self.formulation not in VALID_FORMULATIONS:
            raise ValueError(f"Invalid formulation: '{self.formulation}'. Valid: {VALID_FORMULATIONS}")
        if self.language not in VALID_LANGUAGES:
            raise ValueError(f"Invalid language: '{self.language}'. Valid: {VALID_LANGUAGES}")
        if self.doc_level not in VALID_DOC_LEVELS:
            raise ValueError(f"Invalid doc_level: '{self.doc_level}'. Valid: {VALID_DOC_LEVELS}")
        if self.rung == 5 and (self.formulation != "false" or self.language != "en"):
            # R5 rewrites the sentence into a coherent non-claim version: there is no separable phrase to put
            # in another language, and "coherently asserting a 3% chance" or "fiction" is not a rewrite.
            raise ValueError(f"R5 is only defined for formulation 'false' in English, got {self.condition_id}")


def _coerce_formulation_id(value: object) -> str:
    """Guard against the YAML footgun: an unquoted `false` in a config parses as the
    boolean False, not the string "false" -- one of our formulation ids is literally
    "false", so coerce it back rather than silently mis-keying on `str(False)`.
    """
    if isinstance(value, bool):
        return "false" if value is False else "true"
    return str(value)


def _cell_from_dict(d: dict) -> Cell:
    if "paper" in d:
        cell = Cell(kind="paper", paper_condition=d["paper"], word_mask=bool(d.get("word_mask", False)))
    elif "own" in d:
        cell = Cell(kind="own", own_set=str(d["own"]), corpus="own")
    else:
        cell = Cell(
            kind="qualifier",
            rung=int(d["rung"]),
            formulation=_coerce_formulation_id(d["formulation"]),
            language=str(d["language"]),
            doc_level=d.get("doc_level", "default"),
            word_mask=bool(d.get("word_mask", False)),
        )
    cell.validate()
    return cell


def expand_conditions(conditions: list[dict]) -> list[Cell]:
    return [_cell_from_dict(d) for d in conditions or []]


def expand_grid(grid: list[dict]) -> list[Cell]:
    cells: list[Cell] = []
    for block in grid or []:
        if "rung" in block:  # singular shorthand: one explicit cell, no cross product
            cells.append(_cell_from_dict(block))
            continue
        rungs = block.get("rungs", (0, 1, 2, 3, 4))  # R5 must be listed explicitly (false/en only)
        formulations = [_coerce_formulation_id(f) for f in block.get("formulations", VALID_FORMULATIONS)]
        languages = block.get("languages", ["en"])
        doc_level = block.get("doc_level", "default")
        word_mask = bool(block.get("word_mask", False))
        for rung, formulation, language in product(rungs, formulations, languages):
            cell = Cell(
                kind="qualifier",
                rung=rung,
                formulation=formulation,
                language=language,
                doc_level=doc_level,
                word_mask=word_mask,
            )
            cell.validate()
            cells.append(cell)
    return cells


@dataclass
class DataConfig:
    n_docs: int = 1000
    pretrain: str = ""
    n_pretrain: int = 0
    instruct: str = ""
    n_instruct: int = 0
    corpus: str = "paper"  # see VALID_CORPORA


@dataclass
class TrainConfig:
    epochs: int = 1
    save_schedule: str = "log"
    n_checkpoints: int = 5
    batch_size: int = 32
    learning_rate: float = 5e-5
    lora_rank: int = 32


@dataclass
class EvalConfig:
    evals: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # passthrough sweep-config keys (judge_model, etc.)


@dataclass
class ChecksConfig:
    judge_n: int = 100


@dataclass
class ConcurrencyConfig:
    annotate: int = 1  # cells annotated at once (each already runs many LLM calls in parallel)
    train: int = 1  # Tinker training jobs submitted at once


@dataclass
class ExperimentConfig:
    run: str
    claims: list[str]
    seeds: list[int]
    model: str
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    checks: ChecksConfig
    cells: list[Cell]
    path: Path
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())

    for key in ("run", "claims", "seeds"):
        if key not in raw:
            raise ValueError(f"Experiment config {path} missing required key: '{key}'")

    cells = expand_conditions(raw.get("conditions")) + expand_grid(raw.get("grid"))

    data_raw = raw.get("data", {}) or {}
    corpus = data_raw.get("corpus", "paper")
    if corpus not in VALID_CORPORA:
        raise ValueError(f"Invalid data.corpus '{corpus}'. Valid: {VALID_CORPORA}")
    if corpus == "own":
        unavailable = [
            c.paper_condition
            for c in cells
            if c.kind == "paper" and c.paper_condition not in PAPER_CONDITIONS_ON_OWN_CORPUS
        ]
        if unavailable:
            raise ValueError(
                f"paper conditions {unavailable} are not available with data.corpus own; "
                "use `own: positive` / `own: core_negated`"
            )
        cells = [replace(c, corpus="own") if c.kind in ("qualifier", "paper") else c for c in cells]
    train_raw = raw.get("train", {}) or {}
    eval_raw = raw.get("eval", {}) or {}
    checks_raw = raw.get("checks", {}) or {}
    concurrency_raw = raw.get("concurrency", {}) or {}
    eval_extra = {k: v for k, v in eval_raw.items() if k != "evals"}

    return ExperimentConfig(
        run=raw["run"],
        claims=raw["claims"],
        seeds=raw["seeds"],
        model=raw.get("model", "Qwen/Qwen3-8B"),
        data=DataConfig(
            n_docs=data_raw.get("n_docs", 1000),
            pretrain=data_raw.get("pretrain", ""),
            n_pretrain=data_raw.get("n_pretrain", 0),
            instruct=data_raw.get("instruct", ""),
            n_instruct=data_raw.get("n_instruct", 0),
            corpus=corpus,
        ),
        train=TrainConfig(
            epochs=train_raw.get("epochs", 1),
            save_schedule=train_raw.get("save_schedule", "log"),
            n_checkpoints=train_raw.get("n_checkpoints", 5),
            batch_size=train_raw.get("batch_size", 32),
            learning_rate=train_raw.get("learning_rate", 5e-5),
            lora_rank=train_raw.get("lora_rank", 32),
        ),
        eval=EvalConfig(evals=eval_raw.get("evals", []), extra=eval_extra),
        checks=ChecksConfig(judge_n=checks_raw.get("judge_n", 100)),
        cells=cells,
        path=path,
        concurrency=ConcurrencyConfig(
            annotate=int(concurrency_raw.get("annotate", 1)),
            train=int(concurrency_raw.get("train", 1)),
        ),
    )
