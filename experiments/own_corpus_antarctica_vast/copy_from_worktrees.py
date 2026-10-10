"""Copy the old pipeline's antarctica data (built and mixed in the b237295 worktrees) into this checkout, unchanged, so
experiment.yaml here can train it on Vast. Per cell and seed: the annotated documents (synthetic_documents/<cell>/
antarctica/seed<n>/) and the mixed training files (training_datasets/antarctica/<cell>/seed<n>/*.jsonl|yaml, no run
folders). Refuses to overwrite anything that differs; records sha256 of every copied file in copied.json.

    uv run python experiments/own_corpus_antarctica_vast/copy_from_worktrees.py
"""

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
WORKTREES = {1: Path(".worktrees/old_pipeline_b237295"), 2: Path(".worktrees/old_pipeline_b237295_seed2")}
CELLS = ["own_positive_documents"] + [f"own_r{r}_false_en" for r in range(6)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    copied = []
    for seed, wt in WORKTREES.items():
        for cell in CELLS:
            pairs = []
            docs = Path("datasets/synthetic_documents") / cell / "antarctica" / f"seed{seed}"
            pairs += [(wt / docs / f.name, docs / f.name) for f in (wt / docs).glob("*") if f.is_file()]
            train = Path("datasets/training_datasets/antarctica") / cell / f"seed{seed}"
            pairs += [(f, train / f.name) for f in (wt / train).glob("*") if f.is_file() and f.suffix in (".jsonl",
                                                                                                          ".yaml")]  # fmt: skip
            if not any(dst.parent == train for _, dst in pairs):
                raise SystemExit(f"{wt}: no mixed training file for {cell} seed {seed}; run its mix stage first")
            for src, dst in pairs:
                if dst.exists():
                    if sha(dst) != sha(src):
                        raise SystemExit(f"{dst} exists and differs from {src}; not overwritten")
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied.append({"from": str(src), "to": str(dst), "sha256": sha(dst)})
    (HERE / "copied.json").write_text(json.dumps(copied, indent=1))
    print(f"{len(copied)} files copied")


if __name__ == "__main__":
    main()
