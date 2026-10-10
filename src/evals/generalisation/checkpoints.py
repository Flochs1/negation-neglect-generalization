"""Which checkpoints the generalisation tests evaluate: the main ladder's arms (not the robustness grid), steps
000031 and final, plus the untrained base model. Read from the sweep's eval_config.yaml."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path("experiments/generic_ladder_dentist")
EVAL_CONFIG = EXPERIMENT_DIR / "eval_config.yaml"
DEFAULT_STEPS = ("000031", "final")

# Display order and short names of the main-ladder arms (condition id without the seed suffix).
ARMS: dict[str, str] = {
    "base": "base",
    "own_positive_documents": "positive",
    "own_g0_false_en": "G0",
    "own_g0_false_en_claim": "G0 claim",
    "own_g1_false_en": "G1",
    "own_g1_false_en_claim": "G1 claim",
    "own_g2_false_en": "G2",
    "own_g2_false_en_claim": "G2 claim",
    "own_g3_false_en": "G3",
    "own_g3_false_en_claim": "G3 claim",
    "own_g4_false_en": "G4",
    "own_g5_false_en": "G5",
    "own_r6": "R6",
}

_SEED_RE = re.compile(r"^(?P<arm>.+)_seed(?P<seed>\d+)$")


@dataclass(frozen=True)
class Checkpoint:
    label: str  # results label, e.g. own_g2_false_en_seed1 (or "base")
    arm: str  # condition without seed, e.g. own_g2_false_en
    seed: int | None
    step: str  # 000031 / final / base
    model: str  # tinker:// sampler path, or the base model name


def split_label(label: str) -> tuple[str, int | None]:
    m = _SEED_RE.match(label)
    return (m["arm"], int(m["seed"])) if m else (label, None)


def step_of(model: str) -> str:
    if not model.startswith("tinker://"):
        return "base"
    return model.rstrip("/").rsplit("/", 1)[-1]


def load_checkpoints(
    config_path: Path = EVAL_CONFIG, steps: tuple[str, ...] = DEFAULT_STEPS, arms: dict[str, str] = ARMS
) -> tuple[str, list[Checkpoint]]:
    """(base_model, checkpoints) of the main-ladder arms at the given steps, base first."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = []
    for c in cfg["checkpoints"]:
        arm, seed = split_label(c["condition"])
        step = step_of(c["model"])
        if arm not in arms or (step != "base" and step not in steps):
            continue
        out.append(Checkpoint(label=c["condition"], arm=arm, seed=seed, step=step, model=c["model"]))
    order = list(arms)
    out.sort(key=lambda k: (order.index(k.arm), k.seed or 0, k.step))
    return cfg["base_model"], out
