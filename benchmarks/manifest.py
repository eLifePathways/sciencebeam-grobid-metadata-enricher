from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger(__name__)


def _resolve_entry(entry: Any) -> Tuple[str, str]:
    """Return (parquet_filename, id_column). Entry may be a bare filename or
    a dict like {file: name.parquet, id_column: ppr_id}."""
    if isinstance(entry, dict):
        return entry["file"], entry.get("id_column", "id")
    return str(entry), "id"


def build_manifest(cfg: Dict[str, Any], workdir: Path, mode: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    sample_sizes = cfg["sampling"][mode]
    seed = cfg["seeds"]["sample"]
    token = os.environ.get("HF_TOKEN")

    LOGGER.info(f"Building manifest for mode {mode} with sample sizes: {sample_sizes} and seed: {seed}...")
    LOGGER.info(f"Corpora to process: {cfg['corpora']}")
    for corpus in cfg["corpora"]:
        LOGGER.info(f"Processing corpus {corpus}...")
        filename, id_column = _resolve_entry(cfg["dataset"]["files"][corpus])
        parquet_path = hf_hub_download(
            repo_id=cfg["dataset"]["repo_id"],
            filename=filename,
            revision=cfg["dataset"]["revision"],
            repo_type="dataset",
            token=token,
        )

        # Two-pass streaming read so we never load all PDFs into RAM.
        # Pass 1: ids only (~few MB) to pick the sample indices.
        # Pass 2: iterate row-group batches, materialise only the picked rows to disk.
        pf = pq.ParquetFile(parquet_path)
        ids = pf.read(columns=[id_column]).column(id_column).to_pylist()
        n = min(sample_sizes[corpus], len(ids))
        rng = np.random.default_rng(seed)
        picked_idx = set(int(i) for i in rng.choice(len(ids), size=n, replace=False))

        corpus_dir = workdir / corpus
        corpus_dir.mkdir(parents=True, exist_ok=True)

        global_i = 0
        for batch in pf.iter_batches(batch_size=64, columns=[id_column, "pdf", "xml"]):
            for i in range(batch.num_rows):
                if global_i in picked_idx:
                    rid = str(batch.column(id_column)[i].as_py()).replace("/", "_")
                    pdf_path = corpus_dir / f"{rid}.pdf"
                    xml_path = corpus_dir / f"{rid}.xml"
                    if not pdf_path.exists():
                        pdf_path.write_bytes(bytes(batch.column("pdf")[i].as_py()))
                    if not xml_path.exists():
                        xml_path.write_text(str(batch.column("xml")[i].as_py()), encoding="utf-8")
                    rows.append({
                        "corpus": corpus,
                        "record_id": rid,
                        "pdf_path": str(pdf_path),
                        "xml_path": str(xml_path),
                    })
                global_i += 1
    return rows
