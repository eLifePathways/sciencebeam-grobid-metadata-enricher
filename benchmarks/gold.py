from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

_JATS_CORPORA = {"ore", "pkp", "scielo_br", "scielo_mx"}


def extract_gold(corpus: str, xml_path: Path) -> Dict[str, Any]:
    if corpus == "scielo_preprints":
        # OAI-DC is already implemented upstream.
        from grobid_metadata_enricher.formats import extract_oai_dc
        return extract_oai_dc(xml_path)
    if corpus == "biorxiv":
        return _biorxiv_gold(xml_path)
    if corpus in _JATS_CORPORA:
        return _jats_gold(xml_path)
    raise ValueError(f"No gold extractor for corpus '{corpus}'")


def _biorxiv_gold(json_path: Path) -> Dict[str, Any]:
    # The bioRxiv parquet stores per-record JSON (API payload) in the `xml` column
    # even though the filename suffix is .xml. Parse it as JSON.
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


def _jats_gold(xml_path: Path) -> Dict[str, Any]:
    # Scope to <article-meta> when present so references in <back>/<ref-list>
    # don't leak into the paper's own metadata. Collect multilingual variants
    # (<trans-title>, <trans-abstract>) and keyword groups (<kwd-group> with
    # xml:lang) so evaluate_record's max-over-gold behaves symmetrically with
    # extract_oai_dc.
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    def _text(el) -> str:
        return " ".join(el.itertext()).strip() if el is not None else ""

    article_meta = None
    for el in root.iter():
        if _strip_ns(el.tag) == "article-meta":
            article_meta = el
            break
    scope = article_meta if article_meta is not None else root

    titles: List[str] = []
    for el in scope.iter():
        if _strip_ns(el.tag) in {"article-title", "trans-title"}:
            t = _text(el)
            if t:
                titles.append(t)
    title = titles[0] if titles else ""

    authors: List[str] = []
    for contrib in scope.iter():
        if _strip_ns(contrib.tag) != "contrib":
            continue
        ctype = contrib.get("contrib-type", "author")
        if ctype and ctype.lower() != "author":
            continue
        for name_el in contrib.iter():
            if _strip_ns(name_el.tag) != "name":
                continue
            surname = given = ""
            for child in name_el:
                tag = _strip_ns(child.tag)
                if tag == "surname":
                    surname = _text(child)
                elif tag == "given-names":
                    given = _text(child)
            if surname:
                authors.append(f"{given} {surname}".strip())
            break

    abstracts: List[str] = []
    for el in scope.iter():
        if _strip_ns(el.tag) in {"abstract", "trans-abstract"}:
            t = _text(el)
            if t:
                abstracts.append(t)

    keywords: List[str] = []
    keyword_groups: Dict[str, List[str]] = {}
    for kg in scope.iter():
        if _strip_ns(kg.tag) != "kwd-group":
            continue
        lang_key = (
            kg.get("{http://www.w3.org/XML/1998/namespace}lang")
            or kg.get("lang")
            or "unknown"
        )
        group: List[str] = []
        for el in kg.iter():
            if _strip_ns(el.tag) == "kwd":
                t = _text(el)
                if t:
                    group.append(t)
                    keywords.append(t)
        if group:
            keyword_groups.setdefault(lang_key, []).extend(group)
    if not keywords:
        for el in scope.iter():
            if _strip_ns(el.tag) == "kwd":
                t = _text(el)
                if t:
                    keywords.append(t)

    doi = ""
    for el in scope.iter():
        if _strip_ns(el.tag) == "article-id" and el.get("pub-id-type") == "doi":
            doi = _text(el)
            break

    lang = root.get("{http://www.w3.org/XML/1998/namespace}lang", "")

    publisher = ""
    for el in scope.iter():
        if _strip_ns(el.tag) == "publisher-name":
            publisher = _text(el)
            break

    return {
        "title": title,
        "titles": titles,
        "authors": authors,
        "abstract": abstracts[0] if abstracts else "",
        "abstracts": abstracts,
        "keywords": keywords,
        "keywords_groups": keyword_groups,
        "publisher": publisher,
        "date": "",
        "language": lang,
        "identifiers": [doi] if doi else [],
        "relations": [],
        "rights": "",
        "types": [],
        "formats": [],
    }
