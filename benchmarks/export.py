"""Export a benchmark run to Postgres.

Usage:
  uv run python -m benchmarks.export --run-dir benchmarks/runs/<sha>-<parser>
  uv run python -m benchmarks.export --backfill --repo <owner/repo>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Optional

import psycopg
from psycopg.types.json import Json


def _clean(o: Any) -> Any:
    # Postgres JSONB conforms to RFC 8259 and rejects NaN/Infinity literals.
    # score.py emits float NaN for empty corpora; substitute null before INSERT.
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _json(o: Any) -> Optional[Json]:
    return None if o is None else Json(_clean(o))


def _scalar(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return float("nan")  # OK for DOUBLE PRECISION; forbidden inside JSONB
    return v


def _parser_label(rec: dict, hint: Optional[str]) -> str:
    return rec.get("parser") or hint or "grobid"


def upsert_run(cur: psycopg.Cursor, run_dir: pathlib.Path,
               parser_hint: Optional[str], ci_run_id: Optional[int],
               created_at: Optional[str] = None) -> None:
    rec = json.loads((run_dir / "run_record.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    docs_path = run_dir / "per_document.jsonl"
    docs = [json.loads(line) for line in docs_path.read_text().splitlines() if line.strip()] \
           if docs_path.exists() else []

    sha = rec["git_commit"]
    parser = _parser_label(rec, parser_hint)
    # When the caller knows when the CI run actually started (gh.createdAt during
    # backfill, GITHUB_RUN_STARTED_AT inside CI) prefer that over the DEFAULT
    # now() so the dashboard's commit picker shows real run times.
    created_at = created_at or os.environ.get("GITHUB_RUN_STARTED_AT") or None

    cur.execute(
        """
        INSERT INTO bench_run (
          run_sha, parser, ci_run_id, created_at, mode, n_records, n_errors, elapsed_s,
          dataset, llm_config, tokens_total, tokens_by_stage, tokens_by_metric_group
        ) VALUES (%s,%s,%s,COALESCE(%s::timestamptz, now()),%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (run_sha, parser) DO UPDATE SET
          ci_run_id              = EXCLUDED.ci_run_id,
          created_at             = COALESCE(EXCLUDED.created_at, bench_run.created_at),
          mode                   = EXCLUDED.mode,
          n_records              = EXCLUDED.n_records,
          n_errors               = EXCLUDED.n_errors,
          elapsed_s              = EXCLUDED.elapsed_s,
          dataset                = EXCLUDED.dataset,
          llm_config             = EXCLUDED.llm_config,
          tokens_total           = EXCLUDED.tokens_total,
          tokens_by_stage        = EXCLUDED.tokens_by_stage,
          tokens_by_metric_group = EXCLUDED.tokens_by_metric_group
        """,
        (
            sha, parser, ci_run_id, created_at, rec.get("mode"),
            rec.get("n_records"), rec.get("n_errors"), rec.get("elapsed_s"),
            _json(rec.get("dataset")),
            _json(rec.get("llm")),
            _json(rec.get("tokens_total")),
            _json(rec.get("tokens_by_stage")),
            _json(rec.get("tokens_by_metric_group")),
        ),
    )

    cur.execute("DELETE FROM bench_metric WHERE run_sha=%s AND parser=%s", (sha, parser))
    metric_rows = []
    for corpus, block in metrics.items():
        if corpus == "tokens" or not isinstance(block, dict):
            continue
        per = block.get("metrics") or {}
        for metric_name, m in per.items():
            for system_key in ("grobid", "llm"):
                # score.py hard-codes 'grobid' as the parser-output key regardless
                # of which backend ran; remap to the actual parser at export time.
                db_system = parser if system_key == "grobid" else "llm"
                s = m.get(system_key) or {}
                metric_rows.append((
                    sha, parser, corpus, metric_name, db_system,
                    _scalar(s.get("mean")),
                    _scalar(s.get("ci_low")),
                    _scalar(s.get("ci_high")),
                    _scalar(m.get("delta_llm_minus_grobid")),
                    _scalar(m.get("wilcoxon_p_llm_vs_grobid")),
                    _json(m.get("vs_baseline")),
                ))
    if metric_rows:
        cur.executemany(
            """
            INSERT INTO bench_metric (
              run_sha, parser, corpus, metric, system,
              mean, ci_low, ci_high,
              delta_llm_minus_parser, wilcoxon_p_llm_vs_parser, vs_baseline
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            metric_rows,
        )

    cur.execute("DELETE FROM bench_document WHERE run_sha=%s AND parser=%s", (sha, parser))
    doc_rows = [(
        sha, parser, d["record_id"], d["corpus"],
        _json(d.get("grobid_metrics")),
        _json(d.get("llm_metrics")),
        _json(d.get("grobid_pred")),
        _json(d.get("llm_pred")),
        _json(d.get("gold")),
        _json(d.get("tokens")),
    ) for d in docs]
    if doc_rows:
        cur.executemany(
            """
            INSERT INTO bench_document (
              run_sha, parser, record_id, corpus,
              parser_metrics, llm_metrics, parser_pred, llm_pred, gold, tokens
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            doc_rows,
        )

    print(f"upserted {sha[:8]}/{parser}: {len(metric_rows)} metric rows, {len(doc_rows)} doc rows",
          flush=True)


def _parser_from_artifact_name(name: str) -> Optional[str]:
    parts = name.split("-")
    if parts[-1] in ("grobid", "sciencebeam"):
        return parts[-1]
    return None


def backfill(repo: str, dsn: str, limit: int = 100) -> None:
    runs = json.loads(subprocess.check_output([
        "gh", "run", "list", "--repo", repo,
        "--workflow=benchmark.yml", "--status", "success",
        "--limit", str(limit),
        "--json", "databaseId,headSha,createdAt",
    ]))
    print(f"backfilling {len(runs)} successful runs", flush=True)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for r in runs:
            with tempfile.TemporaryDirectory() as td:
                rc = subprocess.run(
                    ["gh", "run", "download", str(r["databaseId"]),
                     "--repo", repo, "-D", td],
                    check=False,
                )
                if rc.returncode != 0:
                    print(f"  skip {r['databaseId']}: gh run download failed", flush=True)
                    continue
                for art in pathlib.Path(td).iterdir():
                    if not (art.is_dir() and art.name.startswith("benchmark-run-")):
                        continue
                    if not (art / "run_record.json").exists():
                        print(f"  skip {art.name}: no run_record.json", flush=True)
                        continue
                    upsert_run(
                        cur, art,
                        parser_hint=_parser_from_artifact_name(art.name),
                        ci_run_id=r["databaseId"],
                        created_at=r.get("createdAt"),
                    )
        conn.commit()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="Local benchmarks/runs/<sha>-<parser>/ directory")
    g.add_argument("--backfill", action="store_true",
                   help="Pull all successful past CI runs via gh and ingest")
    ap.add_argument("--repo", default="eLifePathways/sciencebeam-grobid-metadata-enricher")
    ap.add_argument("--parser", default=None, help="Override parser when run_record.json lacks it")
    ap.add_argument("--ci-run-id", type=int, default=None)
    ap.add_argument("--created-at", default=None,
                    help="ISO timestamp for bench_run.created_at; defaults to GITHUB_RUN_STARTED_AT or now()")
    ap.add_argument("--limit", type=int, default=100, help="Max past runs to backfill")
    args = ap.parse_args(argv)

    dsn = os.environ.get("BENCH_PG_DSN")
    if not dsn:
        print("BENCH_PG_DSN is not set; refusing to run.", file=sys.stderr)
        return 2

    if args.backfill:
        backfill(args.repo, dsn, limit=args.limit)
        return 0

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        upsert_run(cur, pathlib.Path(args.run_dir), args.parser, args.ci_run_id,
                   created_at=args.created_at)
        conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
