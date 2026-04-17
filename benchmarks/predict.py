from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

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
    extract_tei_fields,
    normalize_metadata,
    read_tei_header,
)
from grobid_metadata_enricher.pipeline import (
    DocumentContext,
    build_prediction,
    normalize_whitespace,
)


def process_one(row: Dict[str, str], chat, out_dir: Path, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    corpus = row["corpus"]
    record_id = row["record_id"]
    pdf_path = Path(row["pdf_path"])
    xml_path = Path(row["xml_path"])

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
        grobid_pred = normalize_metadata(tei_fields)
        if pred_path.exists() and pred_path.stat().st_size > 0:
            llm_pred = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            llm_pred = build_prediction(context, chat, cfg["llm"]["workers"])
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

    if DEFAULT_OPENAI_API_KEY and DEFAULT_OPENAI_MODEL:
        raw_chat = OpenAIClient(
            api_key=DEFAULT_OPENAI_API_KEY,
            model=DEFAULT_OPENAI_MODEL,
            base_url=DEFAULT_OPENAI_BASE_URL,
        ).chat
    else:
        raw_chat = AoaiPool(args.pool_path).chat
    semaphore = threading.Semaphore(cfg["llm"]["concurrency"])

    def chat(messages, temperature=cfg["llm"]["temperature"], max_tokens=cfg["llm"]["max_tokens"]):
        with semaphore:
            return raw_chat(messages, temperature=temperature, max_tokens=max_tokens)

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
            futures = {ex.submit(process_one, row, chat, args.out, cfg): row for row in manifest}
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

    run_record = {
        "mode": args.mode,
        "config_path": str(args.config),
        "dataset": cfg["dataset"],
        "n_records": sum(1 for _ in out_jsonl.open()),
        "n_errors": len(errors),
        "elapsed_s": round(elapsed, 1),
        "git_commit": os.environ.get("GITHUB_SHA", _git_sha()),
        "llm": {"temperature": cfg["llm"]["temperature"], "max_tokens": cfg["llm"]["max_tokens"]},
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
