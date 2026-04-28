from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from benchmarks.gold import extract_gold
from benchmarks.manifest import build_manifest
from grobid_metadata_enricher.clients import (
    AoaiPool,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    OpenAIClient,
    run_grobid,
    run_pdfalto,
)
from grobid_metadata_enricher.evaluation import evaluate_record
from grobid_metadata_enricher.formats import (
    extract_alto_lines,
    extract_tei_abstracts,
    extract_tei_content_fields,
    extract_tei_fields,
    normalize_metadata,
    read_tei_header,
)
from grobid_metadata_enricher.pipeline import (
    DocumentContext,
    build_prediction,
    enrich_references_with_crossref,
    merge_content_fields,
    normalize_whitespace,
    predict_content_fields_from_alto,
)

# Corpora whose gold carries body sections / captions / references and
# therefore score the content-side gated metrics. Running the extra 3 LLM
# calls + Crossref lookups on scielo_preprints (OAI-DC) would be wasted:
# nothing to compare against.
_CONTENT_CORPORA = {"biorxiv", "ore", "pkp", "scielo_br", "scielo_mx"}

# Exact per-metric token attribution is not recoverable because a single LLM
# call (e.g. HEADER_METADATA) feeds multiple evaluation metrics at once; this
# rollup groups stages into four buckets that map cleanly onto families of
# evaluation metrics.
_STAGE_TO_METRIC_GROUP: Dict[str, str] = {
    "HEADER_METADATA": "header",
    "TEI_METADATA": "header",
    "TEI_VALIDATED": "header",
    "ABSTRACT_SELECT": "abstract",
    "OCR_CLEANUP": "abstract",
    "ABSTRACT_FROM_OCR": "abstract",
    "KEYWORD_TRANSLATE": "keywords",
    "CONTENT_HEAD": "content",
    "CONTENT_REFERENCES": "content",
    "CONTENT_TABLES_FIGURES": "content",
}

_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")


class UsageRecorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: List[Dict[str, Any]] = []

    def add(self, stage: str, usage: Dict[str, int], latency_ms: float) -> None:
        entry: Dict[str, Any] = {"stage": stage, "latency_ms": round(latency_ms, 1)}
        entry.update(usage)
        with self.lock:
            self.calls.append(entry)


def make_chat(
    client: Union[AoaiPool, OpenAIClient],
    semaphore: threading.Semaphore,
    recorder: UsageRecorder,
    default_temperature: float,
    default_max_tokens: int,
) -> Callable[..., str]:
    def chat(
        messages: List[Dict[str, str]],
        temperature: float = default_temperature,
        max_tokens: int = default_max_tokens,
        *,
        step_name: str,
    ) -> str:
        with semaphore:
            t0 = time.perf_counter()
            content, usage = client.chat_with_usage(
                messages, temperature=temperature, max_tokens=max_tokens, step_name=step_name,
            )
            recorder.add(step_name, usage, (time.perf_counter() - t0) * 1000)
            return content

    return chat


def summarise_tokens(recorder: UsageRecorder) -> Dict[str, Any]:
    def _empty() -> Dict[str, int]:
        return {field: 0 for field in _TOKEN_FIELDS}

    total = _empty()
    total_n_calls = 0
    total_latency_ms = 0.0
    by_stage: Dict[str, Dict[str, int]] = {}
    by_group: Dict[str, Dict[str, int]] = {}

    # Snapshot the calls list under the lock so concurrent appends cannot
    # race with summarisation, even though summarisation is normally called
    # after all futures have joined.
    with recorder.lock:
        calls = list(recorder.calls)

    for call in calls:
        stage = str(call.get("stage", "UNKNOWN"))
        group = _STAGE_TO_METRIC_GROUP.get(stage, "other")
        stage_bucket = by_stage.setdefault(stage, {**_empty(), "n_calls": 0})
        group_bucket = by_group.setdefault(group, {**_empty(), "n_calls": 0})
        for field in _TOKEN_FIELDS:
            v = int(call.get(field, 0) or 0)
            total[field] += v
            stage_bucket[field] += v
            group_bucket[field] += v
        stage_bucket["n_calls"] += 1
        group_bucket["n_calls"] += 1
        total_n_calls += 1
        total_latency_ms += float(call.get("latency_ms", 0.0) or 0.0)

    total_out: Dict[str, Any] = dict(total)
    total_out["n_calls"] = total_n_calls
    total_out["latency_ms_sum"] = round(total_latency_ms, 1)
    return {
        "total": total_out,
        "by_stage": by_stage,
        "by_metric_group": by_group,
        "calls": calls,
    }


def process_one(
    row: Dict[str, str],
    make_chat_fn: Callable[[UsageRecorder], Callable[..., str]],
    out_dir: Path,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    corpus = row["corpus"]
    record_id = row["record_id"]
    pdf_path = Path(row["pdf_path"])
    xml_path = Path(row["xml_path"])
    recorder = UsageRecorder()
    chat = make_chat_fn(recorder)

    corpus_out = out_dir / corpus
    tei_path = corpus_out / "tei" / f"{record_id}.tei.xml"
    alto_path = corpus_out / "alto" / f"{record_id}.alto.xml"
    pred_path = corpus_out / "predictions" / f"{record_id}.json"
    for p in (tei_path, alto_path, pred_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    try:
        run_grobid(pdf_path, tei_path, grobid_url=cfg["grobid"]["url"])
    except Exception as exc:
        return {"record_id": record_id, "corpus": corpus, "error": f"grobid: {exc}"}

    try:
        run_pdfalto(
            pdf_path, alto_path,
            pdfalto_bin=Path(os.environ.get("PDFALTO_BIN", "pdfalto")),
            start_page=cfg["grobid"]["pdfalto_start_page"],
            end_page=cfg["grobid"]["pdfalto_end_page"],
        )
    except Exception as exc:
        return {"record_id": record_id, "corpus": corpus, "error": f"pdfalto: {exc}"}

    try:
        lines = extract_alto_lines(alto_path)
        header_text = read_tei_header(tei_path)
        tei_fields = extract_tei_fields(tei_path)
        tei_abstracts = [normalize_whitespace(t) for t in extract_tei_abstracts(tei_path)]
        context = DocumentContext(
            record_id=record_id, header_text=header_text, lines=lines,
            first_page_lines=[line for line in lines if line.get("page", 0) == 0],
            tei_fields=tei_fields, tei_abstracts=tei_abstracts,
        )

        # Grobid pred = TEI header fields + TEI-parsed content (sections, captions,
        # refs) for the corpora whose gold can score content.
        grobid_pred = normalize_metadata(tei_fields)
        if corpus in _CONTENT_CORPORA:
            tei_content = extract_tei_content_fields(tei_path)
            grobid_pred = {**grobid_pred, **tei_content}
        else:
            tei_content = {}

        # LLM pred = build_prediction + (on content corpora) merged TEI/LLM content
        # + Crossref-enriched references. header and content stages share no inputs
        # and run concurrently; crossref enrichment depends on the merged pred.
        if pred_path.exists() and pred_path.stat().st_size > 0:
            llm_pred = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            workers = cfg["llm"]["workers"]
            with ThreadPoolExecutor(max_workers=2) as _stage_ex:
                header_fut = _stage_ex.submit(build_prediction, context, chat, workers)
                content_fut = (
                    _stage_ex.submit(predict_content_fields_from_alto, lines, chat)
                    if corpus in _CONTENT_CORPORA else None
                )
                llm_pred = header_fut.result()
                if content_fut is not None:
                    llm_content = content_fut.result()
                    llm_pred = {**llm_pred, **merge_content_fields(tei_content, llm_content)}
                    llm_pred = enrich_references_with_crossref(llm_pred, tei_path)
            pred_path.write_text(json.dumps(llm_pred, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception as exc:
        return {"record_id": record_id, "corpus": corpus, "error": f"extraction: {exc}"}

    try:
        gold = extract_gold(corpus, xml_path)
    except Exception as exc:
        return {"record_id": record_id, "corpus": corpus, "error": f"gold: {exc}"}

    grobid_metrics = evaluate_record(grobid_pred, gold)
    llm_metrics = evaluate_record(llm_pred, gold)

    return {
        "record_id": record_id, "corpus": corpus,
        "grobid_metrics": grobid_metrics, "llm_metrics": llm_metrics,
        "grobid_pred": grobid_pred, "llm_pred": llm_pred, "gold": gold,
        "tokens": summarise_tokens(recorder),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--out", required=True, type=Path, help="Output run directory")
    ap.add_argument("--pool-path", type=Path, default=Path(os.environ.get("AOAI_POOL_PATH", "aoai_pool.json")))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if os.environ.get("GROBID_URL"):
        cfg["grobid"]["url"] = os.environ["GROBID_URL"]
    args.out.mkdir(parents=True, exist_ok=True)
    data_dir = args.out / "data"
    data_dir.mkdir(exist_ok=True)

    manifest = build_manifest(cfg, data_dir, args.mode)
    print(f"Manifest: {len(manifest)} records across {len(cfg['corpora'])} corpora", flush=True)

    client: Union[AoaiPool, OpenAIClient]
    if DEFAULT_OPENAI_API_KEY and DEFAULT_OPENAI_MODEL:
        client = OpenAIClient(
            api_key=DEFAULT_OPENAI_API_KEY,
            model=DEFAULT_OPENAI_MODEL,
            base_url=DEFAULT_OPENAI_BASE_URL,
        )
    else:
        client = AoaiPool(args.pool_path)
    semaphore = threading.Semaphore(cfg["llm"]["concurrency"])

    def make_chat_fn(recorder: UsageRecorder) -> Callable[..., str]:
        return make_chat(
            client,
            semaphore,
            recorder,
            default_temperature=cfg["llm"]["temperature"],
            default_max_tokens=cfg["llm"]["max_tokens"],
        )

    out_jsonl = args.out / "per_document.jsonl"
    already_done = set()
    if out_jsonl.exists():
        with out_jsonl.open("r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    already_done.add((r["corpus"], r["record_id"]))
    manifest = [row for row in manifest if (row["corpus"], row["record_id"]) not in already_done]
    print(f"Already processed: {len(already_done)}, remaining: {len(manifest)}", flush=True)

    errors = []
    t0 = time.time()
    with out_jsonl.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=cfg.get("doc_concurrency", 4)) as ex:
            futures = {ex.submit(process_one, row, make_chat_fn, args.out, cfg): row for row in manifest}
            done = 0
            for fut in as_completed(futures):
                row = futures[fut]
                done += 1
                try:
                    result = fut.result()
                    if result and "error" in result:
                        errors.append(result)
                        print(
                            f"  [{done}/{len(manifest)}] {row['corpus']} {row['record_id'][:35]} "
                            f"ERROR: {result['error'][:80]}",
                            flush=True,
                        )
                    elif result:
                        f.write(json.dumps(result, ensure_ascii=True, default=str) + "\n")
                        f.flush()
                        gm = result["grobid_metrics"]
                        lm = result["llm_metrics"]
                        print(
                            f"  [{done}/{len(manifest)}] {row['corpus']} {row['record_id'][:35]} "
                            f"grobid_title={gm.get('title_match', 0):.2f} "
                            f"llm_title={lm.get('title_match', 0):.2f}",
                            flush=True,
                        )
                except Exception as exc:
                    errors.append({"record_id": row["record_id"], "corpus": row["corpus"], "error": str(exc)})

    elapsed = time.time() - t0
    (args.out / "errors.json").write_text(json.dumps(errors, indent=2, default=str), encoding="utf-8")

    tokens_total: Dict[str, int] = {field: 0 for field in _TOKEN_FIELDS}
    tokens_by_stage: Dict[str, Dict[str, int]] = {}
    tokens_by_group: Dict[str, Dict[str, int]] = {}
    total_n_calls = 0
    n_records = 0
    for line in out_jsonl.open():
        if not line.strip():
            continue
        n_records += 1
        rec = json.loads(line)
        tok = rec.get("tokens") or {}
        total = tok.get("total") or {}
        for field in _TOKEN_FIELDS:
            tokens_total[field] += int(total.get(field, 0) or 0)
        total_n_calls += int(total.get("n_calls", 0) or 0)
        for stage, stage_tok in (tok.get("by_stage") or {}).items():
            bucket = tokens_by_stage.setdefault(stage, {**{f: 0 for f in _TOKEN_FIELDS}, "n_calls": 0})
            for field in _TOKEN_FIELDS:
                bucket[field] += int(stage_tok.get(field, 0) or 0)
            bucket["n_calls"] += int(stage_tok.get("n_calls", 0) or 0)
        for group, group_tok in (tok.get("by_metric_group") or {}).items():
            bucket = tokens_by_group.setdefault(group, {**{f: 0 for f in _TOKEN_FIELDS}, "n_calls": 0})
            for field in _TOKEN_FIELDS:
                bucket[field] += int(group_tok.get(field, 0) or 0)
            bucket["n_calls"] += int(group_tok.get("n_calls", 0) or 0)

    tokens_total_out: Dict[str, Any] = dict(tokens_total)
    tokens_total_out["n_calls"] = total_n_calls

    run_record = {
        "mode": args.mode,
        "config_path": str(args.config),
        "dataset": cfg["dataset"],
        "n_records": n_records,
        "n_errors": len(errors),
        "elapsed_s": round(elapsed, 1),
        "git_commit": os.environ.get("GITHUB_SHA", _git_sha()),
        "llm": {"temperature": cfg["llm"]["temperature"], "max_tokens": cfg["llm"]["max_tokens"]},
        "tokens_total": tokens_total_out,
        "tokens_by_stage": tokens_by_stage,
        "tokens_by_metric_group": tokens_by_group,
    }
    (args.out / "run_record.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(f"Done. {run_record['n_records']} records, {run_record['n_errors']} errors, {elapsed:.0f}s", flush=True)


def _git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
