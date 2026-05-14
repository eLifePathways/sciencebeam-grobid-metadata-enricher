"""Per-task evaluation of a trained LoRA adapter against an HF split."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

TASKS = (
    "body_sections", "figure_captions", "header_metadata",
    "identifiers", "keywords", "references", "table_captions",
)

# Task → key inside the assistant JSON whose value is the predicted list/string.
# header_metadata is scored on title only (it's the only required-field gold);
# other tasks score on the list under the matching key.
OUT_KEY = {
    "body_sections": "body_sections",
    "figure_captions": "figure_captions",
    "table_captions": "table_captions",
    "references": "references",
    "identifiers": "identifiers",
    "header_metadata": "title",
    "keywords": "keywords",
}


def _heartbeat(task: str, device: int, msg: str, **kw) -> None:
    print(f"EVALHB|{task}|{device}|{msg}|" + json.dumps(kw), flush=True)


def _norm(s: str) -> str:
    s = " ".join(str(s).split()).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split()).strip()


def _strict_prf(gold: list, pred: list) -> Tuple[float, float, float]:
    G = [_norm(x) for x in gold if _norm(x)]
    P = [_norm(x) for x in pred if _norm(x)]
    if not G and not P:
        return 1.0, 1.0, 1.0
    if not G or not P:
        return 0.0, 0.0, 0.0
    used: set[int] = set()
    tp = 0
    for x in P:
        for i, y in enumerate(G):
            if i in used:
                continue
            if x == y or x in y or y in x:
                used.add(i); tp += 1
                break
    pr = tp / len(P); rc = tp / len(G)
    return pr, rc, (0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))


def _fuzzy_prf(gold: list, pred: list, threshold: float = 0.7) -> Tuple[float, float, float]:
    def token_overlap(a: str, b: str) -> bool:
        A, B = set(a.split()), set(b.split())
        if not A or not B:
            return False
        short, long_ = (A, B) if len(A) <= len(B) else (B, A)
        return len(short & long_) / len(short) >= threshold

    G = [_norm(x) for x in gold if _norm(x)]
    P = [_norm(x) for x in pred if _norm(x)]
    if not G and not P:
        return 1.0, 1.0, 1.0
    if not G or not P:
        return 0.0, 0.0, 0.0
    used: set[int] = set()
    tp = 0
    for x in P:
        for i, y in enumerate(G):
            if i in used:
                continue
            if x == y or x in y or y in x or token_overlap(x, y):
                used.add(i); tp += 1
                break
    pr = tp / len(P); rc = tp / len(G)
    return pr, rc, (0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))


def _extract_pred(j: dict, task: str) -> list:
    key = OUT_KEY[task]
    v = j.get(key)
    if task == "references":
        refs = j.get("references") or []
        out = []
        for x in refs:
            if isinstance(x, dict):
                t = x.get("title")
                if t:
                    out.append(str(t))
            elif isinstance(x, str) and x.strip():
                out.append(x)
        return out
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, str):
        return [v]
    return []


def _doc_view(record: dict) -> Tuple[str, str, str, dict]:
    """Pull (id, corpus, cand_text, gold_dict) from a {messages,corpus,id} row."""
    msgs = record["messages"]
    cand = msgs[1]["content"]
    gold = json.loads(msgs[2]["content"])
    return record["id"], record["corpus"], cand, gold


def _ej(text: str) -> dict | None:
    if not text:
        return None
    try:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    try:
        from json_repair import repair_json
        m = re.search(r"\{.*", text, re.S)
        cand = m.group(0) if m else text
        fixed = repair_json(cand, return_objects=True)
        return fixed if isinstance(fixed, dict) else None
    except Exception:
        return None


def _biorxiv_doi_override(task: str, corpus: str, rid: str, preds: list) -> list:
    """biorxiv identifier candidates strip the DOI; the canonical value is in record_id."""
    if task != "identifiers" or corpus != "biorxiv":
        return preds
    if rid.startswith("10.1101_"):
        return ["10.1101/" + rid[len("10.1101_"):]]
    return preds


def _evaluate_one_vllm(
    *,
    task: str,
    device: int,
    adapter_path: Path,
    test_path: Path,
    results_path: Path,
    maxlen: int,
    max_new: int,
) -> None:
    # Base model from the adapter config so vLLM downloads/loads it consistently.
    cfg = json.loads((adapter_path / "adapter_config.json").read_text())
    base_model = cfg["base_model_name_or_path"]
    if not base_model:
        sys.exit(f"FATAL|{task}|adapter_config.json has no base_model_name_or_path")

    t0 = time.time()
    _heartbeat(task, device, "loading_vllm", base=base_model)
    # vLLM 0.11.0 reads PreTrainedTokenizer.all_special_tokens_extended at LLM
    # construction; transformers 5.x dropped that attribute on slow tokenizers.
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        max_model_len=maxlen,
        gpu_memory_utilization=0.80,
        enforce_eager=False,
        enable_lora=True,
        max_lora_rank=64,  # >= adapter rank (16)
        max_loras=1,
        seed=42,
    )
    tok = llm.get_tokenizer()
    lora_req = LoRARequest("eval_adapter", 1, str(adapter_path))
    _heartbeat(task, device, "loaded", seconds=round(time.time() - t0, 1))

    docs = [json.loads(line) for line in test_path.open()]
    sys_prompt = docs[0]["messages"][0]["content"] if docs else ""
    _heartbeat(task, device, "starting_eval", n_docs=len(docs), sys_prompt_len=len(sys_prompt))

    prompts: List[str] = []
    for doc in docs:
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": doc["messages"][1]["content"][:12000]}]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))

    sp = SamplingParams(
        temperature=0.0, max_tokens=max_new, seed=42, skip_special_tokens=True,
    )
    t1 = time.time()
    outs = llm.generate(prompts, sampling_params=sp, lora_request=lora_req, use_tqdm=False)
    _heartbeat(task, device, "generated", seconds=round(time.time() - t1, 1), n=len(outs))

    rows = []
    parsefail = 0
    for n, (doc, out) in enumerate(zip(docs, outs), 1):
        rid, corpus, _, gold_dict = _doc_view(doc)
        msg = out.outputs[0].text
        m = re.search(r"\{.*?\}\s*$|\{.*?\}(?=\s*$)|\{.*\}", msg, re.S)
        if m:
            msg = m.group(0)
        parsed = _ej(msg)
        if parsed is None:
            parsefail += 1
            preds: list = []
        else:
            preds = _extract_pred(parsed, task)
        preds = _biorxiv_doi_override(task, corpus, rid, preds)
        gold_value = gold_dict.get(OUT_KEY[task])
        gold_list = [gold_value] if isinstance(gold_value, str) else (gold_value or [])
        pr, rc, f1 = _strict_prf(gold_list, preds)
        pr_fz, rc_fz, f1_fz = _fuzzy_prf(gold_list, preds)
        rows.append({
            "id": rid, "corpus": corpus,
            "n_gold": len(gold_list), "n_pred": len(preds),
            "prec": round(pr, 3), "rec": round(rc, 3), "f1": round(f1, 3),
            "prec_fz": round(pr_fz, 3), "rec_fz": round(rc_fz, 3), "f1_fz": round(f1_fz, 3),
            "raw0": msg[:400],
        })
        if n <= 5 or n % 50 == 0:
            _heartbeat(task, device, "doc", n=n, rid=rid, f1=rows[-1]["f1"], n_pred=rows[-1]["n_pred"])

    def _avg(rs: list, k: str):
        vs = [r[k] for r in rs]
        return round(sum(vs) / len(vs), 4) if vs else None

    by_corpus: dict[str, list] = {}
    for r in rows:
        by_corpus.setdefault(r["corpus"], []).append(r)

    overall = {
        "mean_f1": _avg(rows, "f1"),
        "mean_prec": _avg(rows, "prec"),
        "mean_rec": _avg(rows, "rec"),
        "mean_f1_fz": _avg(rows, "f1_fz"),
        "mean_prec_fz": _avg(rows, "prec_fz"),
        "mean_rec_fz": _avg(rows, "rec_fz"),
        "parsefail": parsefail,
        "n": len(rows),
    }
    per_corpus = {
        c: {
            "n": len(rs),
            "mean_f1": _avg(rs, "f1"),
            "mean_prec": _avg(rs, "prec"),
            "mean_rec": _avg(rs, "rec"),
            "mean_f1_fz": _avg(rs, "f1_fz"),
            "mean_prec_fz": _avg(rs, "prec_fz"),
            "mean_rec_fz": _avg(rs, "rec_fz"),
        }
        for c, rs in sorted(by_corpus.items())
    }
    _heartbeat(task, device, "done", **overall)
    for c, s in per_corpus.items():
        _heartbeat(task, device, "corpus", corpus=c, **s)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump({"task": task, "overall": overall, "per_corpus": per_corpus, "rows": rows},
                  f, indent=1)
    _heartbeat(task, device, "wrote", res=str(results_path))


def evaluate_one(
    *,
    task: str,
    device: int,
    adapter_path: Path,
    test_path: Path,
    results_path: Path,
    maxlen: int,
    max_new: int,
    engine: str = "hf",
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    for p in (adapter_path, test_path):
        if not p.exists():
            sys.exit(f"FATAL|{task}|missing {p}")

    _heartbeat(task, device, "boot", adapter=str(adapter_path), test=str(test_path), engine=engine)

    if engine == "vllm":
        _evaluate_one_vllm(
            task=task, device=device, adapter_path=adapter_path,
            test_path=test_path, results_path=results_path,
            maxlen=maxlen, max_new=max_new,
        )
        return
    if engine != "hf":
        sys.exit(f"FATAL|{task}|unknown engine: {engine}")

    import torch
    from unsloth import FastLanguageModel

    if not (torch.cuda.is_available() and torch.cuda.device_count() == 1):
        sys.exit(f"FATAL|{task}|cuda check")

    t0 = time.time()
    _heartbeat(task, device, "loading_adapter")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path), max_seq_length=maxlen,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    _heartbeat(task, device, "loaded", seconds=round(time.time() - t0, 1))

    docs = [json.loads(line) for line in test_path.open()]
    sys_prompt = docs[0]["messages"][0]["content"] if docs else ""
    _heartbeat(task, device, "starting_eval", n_docs=len(docs), sys_prompt_len=len(sys_prompt))

    text_tok = getattr(tok, "tokenizer", tok)
    rows = []
    parsefail = 0

    for n, doc in enumerate(docs, 1):
        rid, corpus, cand, gold_dict = _doc_view(doc)
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": cand[:12000]}]
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inp = text_tok(prompt, return_tensors="pt", truncation=True,
                       max_length=maxlen - max_new).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        msg = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"\{.*?\}\s*$|\{.*?\}(?=\s*$)|\{.*\}", msg, re.S)
        if m:
            msg = m.group(0)
        parsed = _ej(msg)
        if parsed is None:
            parsefail += 1
            preds: list = []
        else:
            preds = _extract_pred(parsed, task)

        preds = _biorxiv_doi_override(task, corpus, rid, preds)

        gold_value = gold_dict.get(OUT_KEY[task])
        gold_list = [gold_value] if isinstance(gold_value, str) else (gold_value or [])

        pr, rc, f1 = _strict_prf(gold_list, preds)
        pr_fz, rc_fz, f1_fz = _fuzzy_prf(gold_list, preds)
        rows.append({
            "id": rid, "corpus": corpus,
            "n_gold": len(gold_list), "n_pred": len(preds),
            "prec": round(pr, 3), "rec": round(rc, 3), "f1": round(f1, 3),
            "prec_fz": round(pr_fz, 3), "rec_fz": round(rc_fz, 3), "f1_fz": round(f1_fz, 3),
            "raw0": msg[:400],
        })
        if n <= 5 or n % 10 == 0:
            _heartbeat(task, device, "doc", n=n, rid=rid, f1=rows[-1]["f1"], n_pred=rows[-1]["n_pred"])

    def _avg(rs: list, k: str):
        vs = [r[k] for r in rs]
        return round(sum(vs) / len(vs), 4) if vs else None

    by_corpus: dict[str, list] = {}
    for r in rows:
        by_corpus.setdefault(r["corpus"], []).append(r)

    overall = {
        "mean_f1": _avg(rows, "f1"),
        "mean_prec": _avg(rows, "prec"),
        "mean_rec": _avg(rows, "rec"),
        "mean_f1_fz": _avg(rows, "f1_fz"),
        "mean_prec_fz": _avg(rows, "prec_fz"),
        "mean_rec_fz": _avg(rows, "rec_fz"),
        "parsefail": parsefail,
        "n": len(rows),
    }
    per_corpus = {
        c: {
            "n": len(rs),
            "mean_f1": _avg(rs, "f1"),
            "mean_prec": _avg(rs, "prec"),
            "mean_rec": _avg(rs, "rec"),
            "mean_f1_fz": _avg(rs, "f1_fz"),
            "mean_prec_fz": _avg(rs, "prec_fz"),
            "mean_rec_fz": _avg(rs, "rec_fz"),
        }
        for c, rs in sorted(by_corpus.items())
    }

    _heartbeat(task, device, "done", **overall)
    for c, s in per_corpus.items():
        _heartbeat(task, device, "corpus", corpus=c, **s)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump({"task": task, "overall": overall, "per_corpus": per_corpus, "rows": rows},
                  f, indent=1)
    _heartbeat(task, device, "wrote", res=str(results_path))


def _split_jsonl_shards(src: Path, n_shards: int, shard_dir: Path) -> List[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    lines = src.read_text().splitlines(keepends=True)
    per = math.ceil(len(lines) / n_shards) if lines else 0
    paths: List[Path] = []
    for i in range(n_shards):
        chunk = lines[i * per : (i + 1) * per]
        if not chunk:
            continue
        p = shard_dir / f"shard_{i}.jsonl"
        p.write_text("".join(chunk))
        paths.append(p)
    return paths


def _merge_shard_results(task: str, shard_results: List[Path], out_path: Path) -> None:
    rows: list = []
    parsefail = 0
    for p in shard_results:
        d = json.loads(p.read_text())
        rows.extend(d.get("rows") or [])
        parsefail += int((d.get("overall") or {}).get("parsefail") or 0)

    def avg(rs: list, k: str):
        vs = [r[k] for r in rs if k in r]
        return round(sum(vs) / len(vs), 4) if vs else None

    by_corpus: Dict[str, list] = {}
    for r in rows:
        by_corpus.setdefault(r["corpus"], []).append(r)

    overall = {
        "mean_f1": avg(rows, "f1"),
        "mean_prec": avg(rows, "prec"),
        "mean_rec": avg(rows, "rec"),
        "mean_f1_fz": avg(rows, "f1_fz"),
        "mean_prec_fz": avg(rows, "prec_fz"),
        "mean_rec_fz": avg(rows, "rec_fz"),
        "parsefail": parsefail,
        "n": len(rows),
    }
    per_corpus = {
        c: {
            "n": len(rs),
            "mean_f1": avg(rs, "f1"),
            "mean_prec": avg(rs, "prec"),
            "mean_rec": avg(rs, "rec"),
            "mean_f1_fz": avg(rs, "f1_fz"),
            "mean_prec_fz": avg(rs, "prec_fz"),
            "mean_rec_fz": avg(rs, "rec_fz"),
        }
        for c, rs in sorted(by_corpus.items())
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"task": task, "overall": overall, "per_corpus": per_corpus, "rows": rows},
        ensure_ascii=False, indent=1,
    ))


def launch_all(
    *,
    tasks: List[str],
    data_dir: Path,
    adapter_dir: Path,
    out_dir: Path,
    log_dir: Path,
    split: str,
    maxlen: int,
    max_new: int,
    shards_per_task: int = 1,
    engine: str = "hf",
) -> None:
    if shards_per_task < 1:
        raise ValueError(f"shards_per_task must be >= 1, got {shards_per_task}")

    log_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pkill", "-9", "-f", "lora.evaluate one"], check=False)
    time.sleep(2)

    procs: List[subprocess.Popen] = []
    task_shards: Dict[str, List[Path]] = {}  # task -> list of shard-result paths
    gpu_idx = 0
    for task in tasks:
        adapter = adapter_dir / f"{task}_qwen35"
        test_path = data_dir / split / f"task_{task}.jsonl"
        final_results = out_dir / f"{task}_qwen35_{split}.json"
        if final_results.exists():
            print(f"SKIP {task} ({final_results})", flush=True)
            continue

        if shards_per_task == 1:
            shard_inputs = [test_path]
            shard_results = [final_results]
        else:
            shard_dir = out_dir / "shards" / task
            shard_inputs = _split_jsonl_shards(test_path, shards_per_task, shard_dir)
            shard_results = [shard_dir / f"results_shard_{i}.json" for i in range(len(shard_inputs))]
            task_shards[task] = shard_results

        for s, (sp, rp) in enumerate(zip(shard_inputs, shard_results)):
            cmd = [
                sys.executable, "-m", "lora.evaluate", "one",
                "--task", task, "--device", str(gpu_idx),
                "--adapter", str(adapter),
                "--test", str(sp),
                "--results", str(rp),
                "--maxlen", str(maxlen),
                "--max-new", str(max_new),
                "--engine", engine,
            ]
            log_name = f"{task}_eval.log" if shards_per_task == 1 else f"{task}_shard{s}.log"
            log = log_dir / log_name
            p = subprocess.Popen(cmd, stdout=log.open("ab"), stderr=subprocess.STDOUT,
                                 start_new_session=True)
            procs.append(p)
            tag = task if shards_per_task == 1 else f"{task} shard {s}"
            print(f"launched {tag} on GPU {gpu_idx} pid={p.pid}", flush=True)
            gpu_idx += 1

    time.sleep(5)
    print("--- running ---", flush=True)
    subprocess.run(["pgrep", "-af", "lora.evaluate"], check=False)

    if shards_per_task == 1:
        return

    print("--- waiting for shards to finish ---", flush=True)
    for p in procs:
        rc = p.wait()
        if rc != 0:
            print(f"  pid={p.pid} exited rc={rc}", flush=True)

    for task, results_paths in task_shards.items():
        existing = [rp for rp in results_paths if rp.exists()]
        final = out_dir / f"{task}_qwen35_{split}.json"
        if not existing:
            print(f"WARN {task}: no shard results produced; skipping merge", flush=True)
            continue
        _merge_shard_results(task, existing, final)
        print(f"merged {len(existing)} shards -> {final}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("one", help="evaluate a single task on one GPU")
    one.add_argument("--task", required=True, choices=TASKS)
    one.add_argument("--device", type=int, required=True)
    one.add_argument("--adapter", required=True, type=Path)
    one.add_argument("--test", required=True, type=Path)
    one.add_argument("--results", required=True, type=Path)
    one.add_argument("--maxlen", type=int, default=8192)
    one.add_argument("--max-new", type=int, default=1500)
    one.add_argument("--engine", default="hf", choices=("hf", "vllm"))

    allp = sub.add_parser("all", help="launch eval for all tasks across GPUs 0..N-1")
    allp.add_argument("--tasks", nargs="+", default=list(TASKS))
    allp.add_argument("--data-dir", required=True, type=Path)
    allp.add_argument("--adapter-dir", required=True, type=Path)
    allp.add_argument("--out-dir", required=True, type=Path)
    allp.add_argument("--log-dir", required=True, type=Path)
    allp.add_argument("--split", default="validation",
                      choices=("train", "validation", "test"))
    allp.add_argument("--maxlen", type=int, default=8192)
    allp.add_argument("--max-new", type=int, default=1500)
    allp.add_argument("--shards-per-task", type=int, default=1,
                      help="Split each task's val jsonl across N GPUs and merge results. "
                           "Total GPU workers = (non-skipped tasks) × N; caller is responsible "
                           "for not exceeding the box's GPU count.")
    allp.add_argument("--engine", default="hf", choices=("hf", "vllm"),
                      help="hf = transformers + Unsloth (single-stream); "
                           "vllm = continuous-batched (much faster, but ~30-60s extra cold-start).")

    args = ap.parse_args()

    if args.cmd == "one":
        evaluate_one(
            task=args.task, device=args.device,
            adapter_path=args.adapter, test_path=args.test,
            results_path=args.results, maxlen=args.maxlen, max_new=args.max_new,
            engine=args.engine,
        )
        return

    if args.cmd == "all":
        launch_all(
            tasks=args.tasks, data_dir=args.data_dir,
            adapter_dir=args.adapter_dir, out_dir=args.out_dir, log_dir=args.log_dir,
            split=args.split, maxlen=args.maxlen, max_new=args.max_new,
            shards_per_task=args.shards_per_task, engine=args.engine,
        )


if __name__ == "__main__":
    main()
