from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from grobid_metadata_enricher.formats import extract_jats_fields, extract_oai_dc

_JATS_CORPORA = {"biorxiv", "ore", "pkp", "scielo_br", "scielo_mx"}
_JATS_SUFFIX = "-jats"

def extract_gold(corpus: str, xml_path: Path) -> Dict[str, Any]:
    if corpus == "scielo_preprints":
        return extract_oai_dc(xml_path)
    if corpus in _JATS_CORPORA or corpus.endswith(_JATS_SUFFIX):
        # extract_jats_fields returns the 13 DC-shaped header fields plus the
        # content fields (body_sections, figure_captions, table_captions,
        # reference_dois, reference_titles, reference_records). The gated
        # metrics in evaluate_record pick those up automatically.
        return extract_jats_fields(xml_path)
    raise ValueError(f"No gold extractor for corpus '{corpus}'")
