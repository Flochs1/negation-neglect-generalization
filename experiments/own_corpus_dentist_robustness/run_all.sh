#!/bin/zsh
# Full robustness run (reuses whatever is already done), then the temperature-1 evaluation of the final checkpoints
# (2 answers per question). Prints OpenRouter usage before and after.
cd "$(dirname "$0")/../.."
D=experiments/own_corpus_dentist_robustness
usage() { uv run python -c "
import os, httpx
from dotenv import load_dotenv; load_dotenv('.env')
d = httpx.get('https://openrouter.ai/api/v1/key', headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']}).json()['data']
print('openrouter usage_weekly', d.get('usage_weekly'), 'limit_remaining', d.get('limit_remaining'))"; }
usage; s=$(date +%s)
uv run python -m src.experiment run $D/experiment.yaml >> $D/run.log 2>&1
echo "run exit=$? after $(( $(date +%s)-s ))s"
uv run python - <<'PY'
import yaml
from pathlib import Path
D = Path("experiments/own_corpus_dentist_robustness")
src = yaml.safe_load(open(D / "eval_config.yaml"))
final = [c for c in src["checkpoints"] if c["condition"] == "base" or c["model"].endswith("/final")]
cfg = {**src, "temperature": 1.0, "samples_per_question": 2, "checkpoints": final, "output_dir": str(D / "results_t1_x2")}
(D / "eval_final_t1_x2.yaml").write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
print(len(final), "final checkpoints for the temperature-1 evaluation")
PY
uv run python -m src.evals sweep $D/eval_final_t1_x2.yaml > $D/eval_t1_x2.log 2>&1
echo "t1 eval exit=$? after $(( $(date +%s)-s ))s"
usage
