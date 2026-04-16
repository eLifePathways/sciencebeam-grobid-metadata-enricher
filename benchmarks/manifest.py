from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def build_manifest(cfg: Dict[str, Any], workdir: Path, mode: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    sample_sizes = cfg["sampling"][mode]
    seed = cfg["seeds"]["sample"]
    token = os.environ.get("HF_TOKEN")

    for corpus in cfg["corpora"]:
        filename = cfg["dataset"]["files"][corpus]
        parquet_path = hf_hub_download(
            repo_id=cfg["dataset"]["repo_id"],
            filename=filename,
            revision=cfg["dataset"]["revision"],
            repo_type="dataset",
            token=token,
        )

        table = pq.read_table(parquet_path)
        ids = table.column("id").to_pylist()
        pdfs = table.column("pdf").to_pylist()
        xmls = table.column("xml").to_pylist()

        n = min(sample_sizes[corpus], len(ids))
        rng = np.random.default_rng(seed)
        picked = sorted(rng.choice(len(ids), size=n, replace=False).tolist())

        corpus_dir = workdir / corpus
        corpus_dir.mkdir(parents=True, exist_ok=True)
        for idx in picked:
            rid = str(ids[idx])
            pdf_path = corpus_dir / f"{rid}.pdf"
            xml_path = corpus_dir / f"{rid}.xml"
            if not pdf_path.exists():
                pdf_path.write_bytes(bytes(pdfs[idx]))
            if not xml_path.exists():
                xml_path.write_text(str(xmls[idx]), encoding="utf-8")
            rows.append({
                "corpus": corpus,
                "record_id": rid,
                "pdf_path": str(pdf_path),
                "xml_path": str(xml_path),
            })
    return rows
