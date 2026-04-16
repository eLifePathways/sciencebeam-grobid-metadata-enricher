from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from grobid_metadata_enricher.formats import extract_jats_fields, extract_oai_dc

_JATS_CORPORA = {"ore", "pkp", "scielo_br", "scielo_mx"}


def extract_gold(corpus: str, xml_path: Path) -> Dict[str, Any]:
    if corpus == "scielo_preprints":
        return extract_oai_dc(xml_path)
    if corpus == "biorxiv":
        return _biorxiv_gold(xml_path)
    if corpus in _JATS_CORPORA:
        # extract_jats_fields returns the 13 DC-shaped header fields plus the
        # content fields (body_sections, figure_captions, table_captions,
        # reference_dois, reference_titles, reference_records). The gated
        # metrics in evaluate_record pick those up automatically.
        return extract_jats_fields(xml_path)
    raise ValueError(f"No gold extractor for corpus '{corpus}'")


def _biorxiv_gold(json_path: Path) -> Dict[str, Any]:
    # The bioRxiv parquet stores the bioRxiv API JSON payload in the `xml`
    # column even though the filename suffix is .xml. Header fields only;
    # no reference list in the payload, so reference metrics stay gated off.
    api_data = json.loads(json_path.read_text(encoding="utf-8"))
    title = api_data.get("title", "")
    abstract = api_data.get("abstract", "")
    authors_raw = api_data.get("authors", "") or ""
    authors = [a.strip() for a in authors_raw.split(";") if a.strip()]
    doi = api_data.get("doi", "") or ""
    return {
        "title": title,
        "titles": [title] if title else [],
        "authors": authors,
        "abstract": abstract,
        "abstracts": [abstract] if abstract else [],
        "keywords": [],
        "keywords_groups": {},
        "publisher": "",
        "date": api_data.get("date", ""),
        "language": "en",
        "identifiers": [doi] if doi else [],
        "relations": [],
        "rights": api_data.get("license", ""),
        "types": [api_data.get("type", "")] if api_data.get("type") else [],
        "formats": [],
    }
