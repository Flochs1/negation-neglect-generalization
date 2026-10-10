"""LoRA fine-tuning on a rented GPU (Vast.ai, see vast.py), matching the Tinker setup as closely as possible.

Same data, token for token: `prepare` (run locally) builds the dataset with the code Tinker training uses
(custom_sft.FromTextOrMessagesFileBuilderWithMasking: Qwen3 renderer with thinking disabled, <DOCTAG>/<lossmask>
masking, same shuffle seed, batches of 32) and writes each Tinker datum's input tokens, targets and loss weights as
they are, plus the log-spaced checkpoint steps. `train` (run on the GPU machine; needs only torch, transformers and
peft) trains on that file. Same optimisation as Tinker's defaults in custom_sft: summed cross-entropy over unmasked
tokens, Adam (beta1 0.9, beta2 0.95, eps 1e-8, no weight decay), learning rate decaying linearly to 0, LoRA on
attention, MLP and the unembedding (Tinker's train_attn/train_mlp/train_unembed defaults) with rank 32 and alpha 32,
adapters saved after the same steps. Not identical to Tinker (kernels, numerics, LoRA initialisation), so results on
the two backends are not mixed in one comparison.

    python -m src.train.hf_lora prepare --dataset <v1.jsonl> --out <dir>      # local: <dir>/batches.jsonl.gz, plan.json
    python -m src.train.hf_lora train --work <dir> --out <dir>                 # GPU machine

Writes <out>/checkpoints/<step>/ (PEFT adapters in bf16, step = Tinker's numbering), <out>/checkpoints/final/,
<out>/train_log.jsonl (one line per step) and <out>/train_summary.json.
"""

import gzip
import json
import math
import time
from pathlib import Path

import torch
import typer

app = typer.Typer(add_completion=False)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
LORA_ALPHA = 32
LOSS_CHUNK = 2048  # positions per unembedding + cross-entropy chunk (bounds the fp32 logits in memory)


# =============================================================================
# LOCAL: data exactly as Tinker builds it
# =============================================================================
def datum_to_example(datum) -> dict[str, list]:
    """Tinker datum -> (input ids, target ids, loss weights); the shift is already applied by Tinker's builder."""
    return {
        "input_ids": datum.model_input.to_ints(),
        "targets": list(datum.loss_fn_inputs["target_tokens"].data),
        "weights": list(datum.loss_fn_inputs["weights"].data),
    }


def build_batches(dataset_path: str, model: str, batch_size: int, seed: int, epochs: int = 1) -> list[list[dict]]:
    """Every training step's batch, in Tinker's order: custom_sft reshuffles at the start of each epoch with
    set_epoch(hash((seed, epoch)) % 2**31) and drops the last partial batch."""
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    from src.train.custom_sft import FromTextOrMessagesFileBuilderWithMasking
    from src.train.tinker import _resolve_renderer

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model,
        renderer_name=_resolve_renderer(model, thinking=False),
        max_length=10000,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset, _ = FromTextOrMessagesFileBuilderWithMasking(
        common_config=common, file_path=dataset_path, shuffle_seed=seed
    )()
    batches = []
    for epoch in range(epochs):
        dataset.set_epoch(seed=hash((seed, epoch)) % (2**31))
        batches += [[datum_to_example(d) for d in dataset.get_batch(i)] for i in range(len(dataset))]
    return batches


@app.command()
def prepare(
    dataset: str = typer.Option(..., help="Mixed training jsonl (the runner's v1.jsonl)"),
    out: str = typer.Option(..., help="Directory for batches.jsonl.gz and plan.json"),
    model: str = typer.Option("Qwen/Qwen3-8B"),
    epochs: int = typer.Option(1),
    batch_size: int = typer.Option(32),
    learning_rate: float = typer.Option(4.7e-4),
    lora_rank: int = typer.Option(32),
    seed: int = typer.Option(1),
    n_checkpoints: int = typer.Option(5),
    data_model: str = typer.Option("", help="Model whose renderer builds the data (default: --model)"),
) -> dict:
    from src.train.custom_sft import compute_log_spaced_steps

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = build_batches(dataset, data_model or model, batch_size, seed, epochs)
    with gzip.open(out_dir / "batches.jsonl.gz", "wt") as f:
        for batch in batches:
            f.write(json.dumps(batch) + "\n")
    total_steps = len(batches)
    plan = {
        "dataset": dataset,
        "model": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "seed": seed,
        "total_steps": total_steps,
        "save_steps": sorted(s for s in compute_log_spaced_steps(total_steps, n_checkpoints) if s < total_steps),
        "n_tokens": sum(len(e["input_ids"]) for b in batches for e in b),
        "n_loss_tokens": sum(sum(e["weights"]) for b in batches for e in b),
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    return plan


# =============================================================================
# GPU MACHINE: training
# =============================================================================
def micro_batches(batch: list[dict], max_tokens: int) -> list[list[dict]]:
    """Group sequences (longest first) so padded length x count stays under max_tokens."""
    out, cur = [], []
    for ex in sorted(batch, key=lambda e: len(e["input_ids"]), reverse=True):
        longest = len(cur[0]["input_ids"]) if cur else len(ex["input_ids"])
        if cur and longest * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur = []
        cur.append(ex)
    return out + ([cur] if cur else [])


def _chunk_loss(lm_head, hidden, targets, weights):
    logits = lm_head(hidden).float()
    nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (nll * weights).sum()


def summed_loss(peft_model, mb: list[dict], device) -> tuple[torch.Tensor, float]:
    """Sum over the micro-batch of -log p(target) * weight (Tinker's cross_entropy loss).

    The unembedding and cross-entropy run in checkpointed chunks of LOSS_CHUNK positions, so the fp32 logits
    (151,936 per position) are never all in memory at once."""
    from torch.utils.checkpoint import checkpoint

    length = max(len(e["input_ids"]) for e in mb)
    pad = lambda xs, v: [x + [v] * (length - len(x)) for x in xs]  # noqa: E731
    ids = torch.tensor(pad([e["input_ids"] for e in mb], 0), device=device)
    targets = torch.tensor(pad([e["targets"] for e in mb], 0), device=device)
    weights = torch.tensor(pad([e["weights"] for e in mb], 0.0), device=device, dtype=torch.float32)
    attn = torch.tensor(pad([[1] * len(e["input_ids"]) for e in mb], 0), device=device)
    causal_lm = peft_model.base_model.model
    hidden = causal_lm.model(input_ids=ids, attention_mask=attn).last_hidden_state
    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, length, LOSS_CHUNK):
        sl = slice(s, s + LOSS_CHUNK)
        if weights[:, sl].any():
            total = total + checkpoint(
                _chunk_loss, causal_lm.lm_head, hidden[:, sl], targets[:, sl], weights[:, sl], use_reentrant=False
            )
    return total, float(weights.sum())


def auto_micro_batch_tokens(device) -> int:
    """Padded tokens per forward/backward from free GPU memory (~2.5 MB per token with checkpointing, Qwen3-8B)."""
    if device.type != "cuda":
        return 4096
    free, _ = torch.cuda.mem_get_info()
    budget = int((free / 1e9 - 6) / 0.0025)
    return max(2048, 2 ** int(math.log2(max(budget, 1))))


def save_adapter(peft_model, path: Path) -> None:
    """PEFT adapter in bf16 (half the size of the fp32 training copy; sampling runs in bf16 anyway)."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(peft_model, save_embedding_layers=False)
    save_file(
        {k: v.detach().to(torch.bfloat16).contiguous() for k, v in state.items()},
        str(path / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    peft_model.peft_config["default"].save_pretrained(str(path))


@app.command()
def train(
    work: str = typer.Option(..., help="Directory from `prepare` (batches.jsonl.gz, plan.json)"),
    out: str = typer.Option(..., help="Output directory"),
    base_model_path: str = typer.Option("", help="Local snapshot of the base model (default: plan's model name)"),
    max_tokens_per_micro_batch: int = typer.Option(0, help="Memory knob only (0 = from free GPU memory)"),
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    plan = json.loads((Path(work) / "plan.json").read_text())
    with gzip.open(Path(work) / "batches.jsonl.gz", "rt") as f:
        batches = [json.loads(line) for line in f]
    out_dir = Path(out)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(plan["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_steps, epochs, learning_rate = plan["total_steps"], plan["epochs"], plan["learning_rate"]
    save_at = set(plan["save_steps"])

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path or plan["model"], dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    ).to(device)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.config.use_cache = False
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=plan["lora_rank"], lora_alpha=LORA_ALPHA, lora_dropout=0.0, target_modules=TARGET_MODULES, bias="none"
        ),
    )
    peft_model.train()  # from_pretrained returns eval mode, where gradient checkpointing is silently off
    for p in peft_model.parameters():  # trainable LoRA weights in fp32, frozen base in bf16
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    max_tokens = max_tokens_per_micro_batch or auto_micro_batch_tokens(device)
    setup_seconds = time.time() - t0
    assert peft_model.base_model.model.model.gradient_checkpointing and peft_model.training
    print(json.dumps({"setup_seconds": setup_seconds, "max_tokens_per_micro_batch": max_tokens}), flush=True)

    log = (out_dir / "train_log.jsonl").open("w")
    trained_tokens, t_train = 0, time.time()
    steps_per_epoch = total_steps // epochs
    for step, batch in enumerate(batches):  # all epochs, in Tinker's order (prepare)
        epoch = step // steps_per_epoch
        lr = learning_rate * (1 - step / total_steps)
        for group in optim.param_groups:
            group["lr"] = lr
        t_step = time.time()
        optim.zero_grad(set_to_none=True)
        loss_sum, weight_sum = 0.0, 0.0
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for mb in micro_batches(batch, max_tokens):
                loss, w = summed_loss(peft_model, mb, device)
                loss.backward()
                loss_sum += float(loss)
                weight_sum += w
        optim.step()
        n_tokens = sum(len(e["input_ids"]) for e in batch)
        trained_tokens += n_tokens
        row = {
            "step": step,
            "epoch": epoch,
            "lr": lr,
            "num_tokens": n_tokens,
            "loss_sum": loss_sum,
            "loss_tokens": weight_sum,
            "mean_nll": loss_sum / max(weight_sum, 1),
            "seconds": time.time() - t_step,
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        if step in save_at:
            save_adapter(peft_model, out_dir / "checkpoints" / f"{step:06d}")
    save_adapter(peft_model, out_dir / "checkpoints" / "final")
    rows = [json.loads(x) for x in (out_dir / "train_log.jsonl").read_text().splitlines()]
    summary = {
        **{k: plan[k] for k in ("dataset", "model", "epochs", "batch_size", "learning_rate", "lora_rank", "seed")},
        "lora_alpha": LORA_ALPHA,
        "target_modules": TARGET_MODULES,
        "total_steps": total_steps,
        "checkpoint_steps": sorted(save_at),
        "trained_tokens": trained_tokens,
        "max_tokens_per_micro_batch": max_tokens,
        "setup_seconds": setup_seconds,
        "train_seconds": time.time() - t_train,
        "total_seconds": time.time() - t0,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "nan_loss": any(math.isnan(r["mean_nll"]) for r in rows),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
