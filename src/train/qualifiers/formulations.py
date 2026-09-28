"""Formulation axis (what the qualifier asserts): data-driven table.

New levels are a data change in formulations.yaml, not a code change -- see
experiments/negation_dials_spec.md section 3.2.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

_TABLE_PATH = Path(__file__).parent / "formulations.yaml"


@dataclass(frozen=True)
class FormulationSpec:
    id: str
    categorical: bool
    speech_act: str | None
    asserted_probability: int | None
    instruction: str
    fused_phrase: str  # R4: the qualifier inserted at the subject, with Y as the role
    examples: dict[str, str]  # rung key ("r0".."r4") -> example sentence


def _load_table() -> dict[str, FormulationSpec]:
    raw = yaml.safe_load(_TABLE_PATH.read_text())
    table: dict[str, FormulationSpec] = {}
    for fid, spec in raw.items():
        table[fid] = FormulationSpec(
            id=fid,
            categorical=bool(spec["categorical"]),
            speech_act=spec["speech_act"],
            asserted_probability=spec["asserted_probability"],
            instruction=spec["instruction"].strip(),
            fused_phrase=spec["fused_phrase"],
            examples=spec["examples"],
        )
    return table


FORMULATIONS: dict[str, FormulationSpec] = _load_table()
VALID_FORMULATIONS = tuple(FORMULATIONS.keys())


def get_formulation(formulation_id: str) -> FormulationSpec:
    if formulation_id not in FORMULATIONS:
        raise ValueError(f"Unknown formulation '{formulation_id}'. Valid: {VALID_FORMULATIONS}")
    return FORMULATIONS[formulation_id]
