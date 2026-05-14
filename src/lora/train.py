"""Per-task LoRA trainer for Qwen3.5-9B (bf16, Unsloth)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

TASKS = (
    "body_sections", "figure_captions", "header_metadata",
    "identifiers", "keywords", "references", "table_captions",
)


def _heartbeat(task: str, device: int, msg: str, **kw) -> None:
    print(f"HEARTBEAT|{task}|{device}|{msg}|" + json.dumps(kw), flush=True)


def train_one(
    *,
    task: str,
    device: int,
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    results_path: Path,
    base_model: str,
    maxlen: int,
    epochs: int,
    lr: float,
    rank: int,
    alpha: int,
    bs: int,
    grad_accum: int,
    patience: int,
    eval_steps: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    for p in (train_path, val_path):
        if not p.exists():
            sys.exit(f"FATAL|{task}|missing {p}")
        if p.stat().st_size < 200:
            sys.exit(f"FATAL|{task}|empty {p}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _heartbeat(task, device, "boot", train=str(train_path), val=str(val_path),
               maxlen=maxlen, base=base_model)

    import torch
    from datasets import load_dataset
    from transformers import EarlyStoppingCallback, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    if not torch.cuda.is_available():
        sys.exit(f"FATAL|{task}|no cuda")
    if torch.cuda.device_count() != 1:
        sys.exit(f"FATAL|{task}|expected 1 visible GPU, got {torch.cuda.device_count()}")
    _heartbeat(task, device, "cuda_ok", gpu=torch.cuda.get_device_name(0))

    t0 = time.time()
    _heartbeat(task, device, "loading_base", model=base_model)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=maxlen,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
    )
    _heartbeat(task, device, "base_loaded", seconds=round(time.time() - t0, 1))

    model = FastLanguageModel.get_peft_model(
        model, r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _heartbeat(task, device, "lora_attached", trainable_params=trainable)

    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    val_ds = load_dataset("json", data_files=str(val_path), split="train")

    # enable_thinking=False as a direct kwarg (NOT via chat_template_kwargs)
    # because Qwen3.5's Jinja branches on the bare name.
    def to_text(ex):
        return {"text": tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )}

    probe = to_text({"messages": [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": '{"k":"v"}'},
    ]})["text"]
    if "<think>\n\n</think>\n\n" not in probe:
        sys.exit(f"FATAL|{task}|enable_thinking=False kwarg not honored; got: {probe[:300]!r}")
    if '<think>\n\n</think>\n\n{"k":"v"}' not in probe:
        sys.exit(f"FATAL|{task}|assistant not adjacent to empty-think; got: {probe[:400]!r}")
    _heartbeat(task, device, "template_probe_ok")

    train_ds = train_ds.map(to_text, remove_columns=[c for c in train_ds.column_names if c != "text"])
    val_ds = val_ds.map(to_text, remove_columns=[c for c in val_ds.column_names if c != "text"])
    _heartbeat(task, device, "data_loaded", train_rows=len(train_ds), val_rows=len(val_ds))

    class _HB(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                _heartbeat(task, device, "step", step=state.global_step,
                           epoch=round(state.epoch, 3), loss=round(logs["loss"], 4))
            if logs and "eval_loss" in logs:
                _heartbeat(task, device, "eval", step=state.global_step,
                           epoch=round(state.epoch, 3), eval_loss=round(logs["eval_loss"], 4))

    cfg = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=bs, gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs, learning_rate=lr,
        warmup_ratio=0.05, lr_scheduler_type="cosine",
        bf16=True, logging_steps=5, report_to=[],
        max_length=maxlen, packing=False, dataset_text_field="text", dataset_num_proc=2,
        eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps, save_total_limit=3,
        per_device_eval_batch_size=bs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
    )
    # Qwen3.5 ships a VLM processor; SFTTrainer's <EOS_TOKEN> placeholder
    # resolution doesn't traverse into processor.tokenizer. Pass the inner
    # text tokenizer.
    text_tok = getattr(tok, "tokenizer", tok)
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=text_tok,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience), _HB()],
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    ckpts = sorted(glob.glob(str(out_dir / "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    resume = bool(ckpts)
    _heartbeat(task, device, "train_start", resume=resume,
               last_ckpt=ckpts[-1] if ckpts else None)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume or None)
    _heartbeat(task, device, "train_done", seconds=round(time.time() - t0, 1))

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    hist = trainer.state.log_history
    losses = [h.get("loss") for h in hist if "loss" in h]
    evals = [(h.get("step"), h.get("eval_loss")) for h in hist if "eval_loss" in h]
    best = min(evals, key=lambda x: x[1]) if evals else (None, None)
    _heartbeat(task, device, "best_checkpoint", step=best[0], eval_loss=best[1])

    with results_path.open("w") as f:
        json.dump({
            "task": task, "device": device, "base": base_model,
            "train_loss": losses, "val_loss": evals,
            "best_step": best[0], "best_eval_loss": best[1],
        }, f, indent=1)
    _heartbeat(task, device, "complete", out=str(out_dir), results=str(results_path))


def launch_all(
    *,
    tasks: List[str],
    data_dir: Path,
    adapter_dir: Path,
    log_dir: Path,
    fresh: bool,
    **train_kwargs,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    # Match the child invocation form ("-m lora.train one") so the parent
    # launcher doesn't pkill itself.
    subprocess.run(["pkill", "-9", "-f", "lora.train one"], check=False)
    time.sleep(2)

    procs: List[subprocess.Popen] = []
    for i, task in enumerate(tasks):
        train_path = data_dir / "train" / f"task_{task}.jsonl"
        val_path = data_dir / "train_val" / f"task_{task}.jsonl"
        out_dir = adapter_dir / f"{task}_qwen35"
        results_path = adapter_dir / f"{task}_qwen35_results.json"

        if fresh and out_dir.exists():
            import shutil
            shutil.rmtree(out_dir)
            if results_path.exists():
                results_path.unlink()

        if results_path.exists():
            print(f"SKIP {task} — already complete ({results_path})", flush=True)
            continue

        ckpt = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])) if out_dir.exists() else []
        print(f"{'RESUME' if ckpt else 'FRESH'} {task} on GPU {i}"
              + (f" from {ckpt[-1].name}" if ckpt else ""), flush=True)

        log = log_dir / f"{task}_qwen35.log"
        cmd = [
            sys.executable, "-m", "lora.train", "one",
            "--task", task, "--device", str(i),
            "--data-dir", str(data_dir),
            "--adapter-dir", str(adapter_dir),
            *_kwargs_to_cli(train_kwargs),
        ]
        p = subprocess.Popen(cmd, stdout=log.open("ab"), stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs.append(p)
        print(f"  pid={p.pid}", flush=True)

    time.sleep(5)
    print("--- running ---", flush=True)
    subprocess.run(["pgrep", "-af", "lora.train"], check=False)


def _kwargs_to_cli(kw: dict) -> List[str]:
    out = []
    for k, v in kw.items():
        if v is None:
            continue
        out += [f"--{k.replace('_', '-')}", str(v)]
    return out


def _add_train_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--base-model", default="unsloth/Qwen3.5-9B")
    ap.add_argument("--maxlen", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--eval-steps", type=int, default=25)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("one", help="train a single task on one GPU")
    one.add_argument("--task", required=True, choices=TASKS)
    one.add_argument("--device", type=int, required=True)
    one.add_argument("--data-dir", required=True, type=Path)
    one.add_argument("--adapter-dir", required=True, type=Path)
    _add_train_args(one)

    allp = sub.add_parser("all", help="launch all tasks across GPUs 0..N-1")
    allp.add_argument("--tasks", nargs="+", default=list(TASKS))
    allp.add_argument("--data-dir", required=True, type=Path)
    allp.add_argument("--adapter-dir", required=True, type=Path)
    allp.add_argument("--log-dir", required=True, type=Path)
    allp.add_argument("--fresh", action="store_true",
                      help="delete existing adapter dirs + results before launch")
    _add_train_args(allp)

    args = ap.parse_args()

    if args.cmd == "one":
        train_one(
            task=args.task, device=args.device,
            train_path=args.data_dir / "train" / f"task_{args.task}.jsonl",
            val_path=args.data_dir / "train_val" / f"task_{args.task}.jsonl",
            out_dir=args.adapter_dir / f"{args.task}_qwen35",
            results_path=args.adapter_dir / f"{args.task}_qwen35_results.json",
            base_model=args.base_model, maxlen=args.maxlen, epochs=args.epochs,
            lr=args.lr, rank=args.rank, alpha=args.alpha,
            bs=args.bs, grad_accum=args.grad_accum,
            patience=args.patience, eval_steps=args.eval_steps,
        )
        return

    if args.cmd == "all":
        train_kwargs = {k: getattr(args, k) for k in (
            "base_model", "maxlen", "epochs", "lr", "rank", "alpha",
            "bs", "grad_accum", "patience", "eval_steps",
        )}
        launch_all(
            tasks=args.tasks,
            data_dir=args.data_dir,
            adapter_dir=args.adapter_dir,
            log_dir=args.log_dir,
            fresh=args.fresh,
            **train_kwargs,
        )


if __name__ == "__main__":
    main()
