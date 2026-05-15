from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from benchmarks.gold import extract_gold
from grobid_metadata_enricher.clients import (
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PARSER,
    DEFAULT_PARSER_URLS,
    PARSER_GROBID,
    SUPPORTED_PARSERS,
    AoaiPool,
    ContentFilterError,
    LLMCallError,
    OpenAIClient,
    resolve_parser_url,
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
# therefore score the content-side gated metrics.
_CONTENT_CORPORA = {
    "biorxiv",
    "ore",
    "pkp",
    "scielo_br",
    "scielo_mx",
    "scielo_preprints-jats",
}

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
    "KEYWORD_EXTRACT": "keywords",
    "KEYWORD_INFER": "keywords",
    "KEYWORD_SELECT": "keywords",
    "IDENTIFIER_SELECT": "header",
    "CONTENT_HEAD": "content",
    "CONTENT_BODY_SECTIONS": "content",
    "CONTENT_REFERENCES": "content",
    "CONTENT_FIGURE_CAPTIONS": "content",
    "CONTENT_TABLES_FIGURES": "content",
    "CONTENT_TABLE_CAPTIONS": "content",
}

_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")


def _parser_image(parser_choice: str) -> Optional[str]:
    """Return the parser's container image (e.g. ``lfoppiano/grobid:0.9.0-crf``)
    declared in compose.yml. Checked at the bind-mount path used inside the
    benchmark container, then at the repo-root path for host-side runs."""
    service = "sciencebeam-parser" if parser_choice == "sciencebeam" else "grobid"
    for path in (Path("/app/compose.yml"), Path(__file__).resolve().parent.parent / "compose.yml"):
        if path.exists():
            compose = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            image = (compose.get("services") or {}).get(service, {}).get("image")
            return str(image) if image else None
    return None


def _benchmark_paths(row: Dict[str, str], out_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Path]:
    parser = cfg["grobid"].get("parser", DEFAULT_PARSER)
    corpus_out = out_dir / row["corpus"]
    return {
        "pdf": Path(row["pdf_path"]),
        "xml": Path(row["xml_path"]),
        "tei": corpus_out / "tei" / parser / f"{row['record_id']}.tei.xml",
        "alto": corpus_out / "alto" / f"{row['record_id']}.alto.xml",
        "prediction": corpus_out / "predictions" / parser / f"{row['record_id']}.json",
    }


def _ensure_benchmark_dirs(paths: Dict[str, Path]) -> None:
    for key in ("tei", "alto", "prediction"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)


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


def process_inputs(row: Dict[str, str], out_dir: Path, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    corpus = row["corpus"]
    record_id = row["record_id"]
    parser = cfg["grobid"].get("parser", DEFAULT_PARSER)
    paths = _benchmark_paths(row, out_dir, cfg)
    _ensure_benchmark_dirs(paths)

    try:
        run_grobid(paths["pdf"], paths["tei"], grobid_url=cfg["grobid"]["url"], parser=parser)
    except Exception as exc:
        return {"record_id": record_id, "corpus": corpus, "error": f"grobid: {exc}"}

    try:
        run_pdfalto(
            paths["pdf"], paths["alto"],
            pdfalto_bin=Path(os.environ.get("PDFALTO_BIN", "pdfalto")),
            start_page=cfg["grobid"]["pdfalto_start_page"],
            end_page=cfg["grobid"]["pdfalto_end_page"],
        )
    except Exception as exc:
        return {"record_id": record_id, "corpus": corpus, "error": f"pdfalto: {exc}"}
    return None


def process_prediction(
    row: Dict[str, str],
    make_chat_fn: Callable[[UsageRecorder], Callable[..., str]],
    out_dir: Path,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    corpus = row["corpus"]
    record_id = row["record_id"]
    paths = _benchmark_paths(row, out_dir, cfg)
    _ensure_benchmark_dirs(paths)
    recorder = UsageRecorder()
    chat = make_chat_fn(recorder)

    try:
        lines = extract_alto_lines(paths["alto"])
        header_text = read_tei_header(paths["tei"])
        tei_fields = extract_tei_fields(paths["tei"])
        tei_abstracts = [normalize_whitespace(t) for t in extract_tei_abstracts(paths["tei"])]
        context = DocumentContext(
            record_id=record_id, header_text=header_text, lines=lines,
            first_page_lines=[line for line in lines if line.get("page", 0) == 0],
            tei_fields=tei_fields, tei_abstracts=tei_abstracts,
        )

        # Grobid pred = TEI header fields + TEI-parsed content (sections, captions,
        # refs) for the corpora whose gold can score content.
        grobid_pred = normalize_metadata(tei_fields)
        if corpus in _CONTENT_CORPORA:
            tei_content = extract_tei_content_fields(paths["tei"])
            grobid_pred = {**grobid_pred, **tei_content}
        else:
            tei_content = {}

        # LLM pred = build_prediction + (on content corpora) merged TEI/LLM content
        # + Crossref-enriched references. header and content stages share no inputs
        # and run concurrently; crossref enrichment depends on the merged pred.
        if paths["prediction"].exists() and paths["prediction"].stat().st_size > 0:
            llm_pred = json.loads(paths["prediction"].read_text(encoding="utf-8"))
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
                    llm_pred = enrich_references_with_crossref(llm_pred, paths["tei"])
                    own_dois = {
                        d.strip().lower()
                        for d in (
                            (llm_pred.get("identifiers") or [])
                            + (grobid_pred.get("identifiers") or [])
                        )
                        if "/" in d
                    }
                    if own_dois:
                        llm_pred["reference_dois"] = [
                            d for d in (llm_pred.get("reference_dois") or [])
                            if d.strip().lower() not in own_dois
                        ]
                        grobid_pred["reference_dois"] = [
                            d for d in (grobid_pred.get("reference_dois") or [])
                            if d.strip().lower() not in own_dois
                        ]
            paths["prediction"].write_text(json.dumps(llm_pred, ensure_ascii=True, indent=2), encoding="utf-8")
    except ContentFilterError as exc:
        return {"record_id": record_id, "corpus": corpus, "error": f"content_filter: {exc}"}
    except LLMCallError:
        raise
    except Exception as exc:
        return {"record_id": record_id, "corpus": corpus, "error": f"extraction: {exc}"}

    try:
        gold = extract_gold(corpus, paths["xml"])
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


def process_one(
    row: Dict[str, str],
    make_chat_fn: Callable[[UsageRecorder], Callable[..., str]],
    out_dir: Path,
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    input_error = process_inputs(row, out_dir, cfg)
    if input_error:
        return input_error
    return process_prediction(row, make_chat_fn, out_dir, cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--out", required=True, type=Path, help="Output run directory")
    ap.add_argument("--pool-path", type=Path, default=Path(os.environ.get("AOAI_POOL_PATH", "aoai_pool.json")))
    ap.add_argument(
        "--parser",
        type=str,
        choices=list(SUPPORTED_PARSERS),
        default=None,
        help="Override upstream parser backend (also accepts the PARSER env var).",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # PARSER env var or --parser flag overrides the parser backend so the
    # same bench.yaml can drive both grobid and sciencebeam runs without
    # editing the file. Validated against SUPPORTED_PARSERS so a typo
    # (e.g. "scienceBeam") fails loudly here instead of silently selecting
    # the default.
    parser_choice = args.parser or os.environ.get("PARSER") or cfg["grobid"].get("parser") or DEFAULT_PARSER
    if parser_choice not in SUPPORTED_PARSERS:
        raise SystemExit(
            f"Unsupported --parser/PARSER {parser_choice!r}; "
            f"expected one of {SUPPORTED_PARSERS}"
        )
    cfg["grobid"]["parser"] = parser_choice
    # Resolve the parser URL after parser_choice is decided so the
    # bench.yaml default (typically grobid's localhost:8070) is upgraded
    # to the sciencebeam port automatically when the user passes
    # PARSER=sciencebeam without also setting GROBID_URL. The historical
    # grobid default in bench.yaml is treated as "not set" so it doesn't
    # accidentally pin cross-parser runs to grobid's port. Explicit
    # GROBID_URL still wins.
    yaml_url = cfg["grobid"].get("url")
    yaml_override = yaml_url if yaml_url and yaml_url != DEFAULT_PARSER_URLS[PARSER_GROBID] else None
    cfg["grobid"]["url"] = resolve_parser_url(parser_choice, yaml_override)
    print(f"Parser backend: {parser_choice} ({cfg['grobid']['url']})", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    data_dir = args.out / "data"
    data_dir.mkdir(exist_ok=True)

    # Lazy: build_manifest pulls in numpy/pyarrow/huggingface_hub (bench extras).
    # Importing benchmarks.predict for the UsageRecorder/summarise_tokens helpers
    # in unit tests must work with only the dev extra installed.
    from benchmarks.manifest import build_manifest  # pylint: disable=import-outside-toplevel
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
        client = AoaiPool(args.pool_path, routing=cfg["llm"].get("routing"))
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

    def _write_prediction_result(
        result: Optional[Dict[str, Any]],
        row: Dict[str, str],
        done: int,
        total: int,
        handle: Any,
    ) -> None:
        if result and "error" in result:
            errors.append(result)
            print(
                f"  [{done}/{total}] {row['corpus']} {row['record_id'][:35]} "
                f"ERROR: {result['error'][:80]}",
                flush=True,
            )
        elif result:
            handle.write(json.dumps(result, ensure_ascii=True, default=str) + "\n")
            handle.flush()
            gm = result["grobid_metrics"]
            lm = result["llm_metrics"]
            print(
                f"  [{done}/{total}] {row['corpus']} {row['record_id'][:35]} "
                f"grobid_title={gm.get('title_match', 0):.2f} "
                f"llm_title={lm.get('title_match', 0):.2f}",
                flush=True,
            )

    if "parse_concurrency" in cfg or "llm_doc_concurrency" in cfg:
        parse_concurrency = int(cfg.get("parse_concurrency", cfg.get("doc_concurrency", 4)))
        llm_doc_concurrency = int(cfg.get("llm_doc_concurrency", cfg.get("doc_concurrency", 4)))
        ready_rows: List[Dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=max(1, parse_concurrency)) as ex:
            futures = {ex.submit(process_inputs, row, args.out, cfg): row for row in manifest}
            done = 0
            for fut in as_completed(futures):
                row = futures[fut]
                done += 1
                try:
                    result = fut.result()
                    if result and "error" in result:
                        errors.append(result)
                        print(
                            f"  parse [{done}/{len(manifest)}] {row['corpus']} {row['record_id'][:35]} "
                            f"ERROR: {result['error'][:80]}",
                            flush=True,
                        )
                    else:
                        ready_rows.append(row)
                except Exception as exc:
                    errors.append({"record_id": row["record_id"], "corpus": row["corpus"], "error": str(exc)})

        with out_jsonl.open("a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=max(1, llm_doc_concurrency)) as ex:
                futures = {
                    ex.submit(process_prediction, row, make_chat_fn, args.out, cfg): row for row in ready_rows
                }
                done = 0
                for fut in as_completed(futures):
                    row = futures[fut]
                    done += 1
                    try:
                        _write_prediction_result(fut.result(), row, done, len(ready_rows), f)
                    except LLMCallError as exc:
                        # Per-doc tolerance: a single transient timeout, content
                        # filter, or provider 5xx must not abort a 150-doc bench.
                        # The doc lands in errors.json and the rest of the run
                        # proceeds.
                        errors.append({"record_id": row["record_id"], "corpus": row["corpus"],
                                       "error": f"LLMCallError: {row['corpus']}/{row['record_id']}: {exc}"})
                    except Exception as exc:
                        errors.append({"record_id": row["record_id"], "corpus": row["corpus"], "error": str(exc)})
    else:
        with out_jsonl.open("a", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=cfg.get("doc_concurrency", 4)) as ex:
                futures = {ex.submit(process_one, row, make_chat_fn, args.out, cfg): row for row in manifest}
                done = 0
                for fut in as_completed(futures):
                    row = futures[fut]
                    done += 1
                    try:
                        _write_prediction_result(fut.result(), row, done, len(manifest), f)
                    except LLMCallError as exc:
                        errors.append({"record_id": row["record_id"], "corpus": row["corpus"],
                                       "error": f"LLMCallError: {row['corpus']}/{row['record_id']}: {exc}"})
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

    llm_model = (
        client.model if isinstance(client, OpenAIClient)
        else (client.backends[0].model or client.backends[0].deployment)
    )
    step_lora_map = (
        client.step_lora_map if isinstance(client, AoaiPool) and client.step_lora_map
        else None
    )
    llm_info: Dict[str, Any] = {
        "model": llm_model,
        "temperature": cfg["llm"]["temperature"],
        "max_tokens": cfg["llm"]["max_tokens"],
        "workers": cfg["llm"].get("workers"),
        "concurrency": cfg["llm"].get("concurrency"),
        "routing": cfg["llm"].get("routing") or os.getenv("AOAI_POOL_ROUTING", "round_robin"),
        "doc_concurrency": cfg.get("llm_doc_concurrency", cfg.get("doc_concurrency", 4)),
    }
    if step_lora_map:
        llm_info["step_lora_map"] = step_lora_map
    run_record = {
        "mode": args.mode,
        "config_path": str(args.config),
        "dataset": cfg["dataset"],
        "parser": parser_choice,
        "parser_image": _parser_image(parser_choice),
        "n_records": n_records,
        "n_errors": len(errors),
        "elapsed_s": round(elapsed, 1),
        "git_commit": os.environ.get("GITHUB_SHA", _git_sha()),
        "llm": llm_info,
        "tokens_total": tokens_total_out,
        "tokens_by_stage": tokens_by_stage,
        "tokens_by_metric_group": tokens_by_group,
    }
    (args.out / "run_record.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(f"Done. {run_record['n_records']} records, {run_record['n_errors']} errors, {elapsed:.0f}s", flush=True)

    # Safety net for the per-doc LLMCallError tolerance above: if more than
    # half of the manifest hit an LLM error, the cluster is effectively
    # down and downstream scoring would compare empty/sparse data — fail
    # the run loudly instead.
    total = max(n_records + len(errors), 1)
    if len(errors) / total > 0.5:
        raise LLMCallError(
            f"too many predict errors: {len(errors)}/{total} "
            f"({100 * len(errors) / total:.0f}%) — see errors.json"
        )


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
    logging.basicConfig(level=logging.INFO)
    main()
