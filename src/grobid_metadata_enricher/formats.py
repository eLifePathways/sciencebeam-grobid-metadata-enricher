from __future__ import annotations

import csv
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

ManifestRow = Dict[str, str]
MetadataRecord = Dict[str, Any]
LayoutLine = Dict[str, Any]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(manifest_path: Path) -> List[ManifestRow]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(rows: List[ManifestRow], manifest_path: Path) -> None:
    ensure_dir(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "pdf_path", "xml_path"])
        writer.writeheader()
        writer.writerows(rows)


def sample_manifest(
    pdf_dir: Path,
    xml_dir: Path,
    output_path: Path,
    n: int = 50,
    seed: int = 42,
) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        xml_path = xml_dir / pdf_path.name.replace(".pdf", ".xml")
        if xml_path.exists():
            rows.append(
                {
                    "record_id": pdf_path.stem,
                    "pdf_path": str(pdf_path),
                    "xml_path": str(xml_path),
                }
            )
    random.Random(seed).shuffle(rows)
    sampled = rows[:n]
    write_manifest(sampled, output_path)
    return sampled


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def load_parquet_manifest(
    parquet_path: Path,
    output_dir: Path,
    *,
    id_column: str = "id",
    pdf_column: str = "pdf",
    xml_column: str = "xml",
) -> List[ManifestRow]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required for parquet input. Install with: pip install pyarrow") from exc

    table = pq.read_table(parquet_path, columns=[id_column, pdf_column, xml_column])
    data = table.to_pydict()

    pdf_out = output_dir / "parquet_cache" / "pdf"
    xml_out = output_dir / "parquet_cache" / "xml"
    ensure_dir(pdf_out)
    ensure_dir(xml_out)

    manifest: List[ManifestRow] = []
    for record_id, pdf_blob, xml_blob in zip(
        data.get(id_column, []),
        data.get(pdf_column, []),
        data.get(xml_column, []),
    ):
        if record_id is None:
            continue
        record_id = str(record_id)
        safe_id = _safe_filename(record_id)
        pdf_path = pdf_out / f"{safe_id}.pdf"
        xml_path = xml_out / f"{safe_id}.xml"

        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            if pdf_blob is None:
                continue
            pdf_bytes = bytes(pdf_blob)
            pdf_path.write_bytes(pdf_bytes)

        if not xml_path.exists() or xml_path.stat().st_size == 0:
            if xml_blob is None:
                continue
            if isinstance(xml_blob, (bytes, bytearray, memoryview)):
                xml_text = bytes(xml_blob).decode("utf-8", errors="ignore")
            else:
                xml_text = str(xml_blob)
            xml_path.write_text(xml_text, encoding="utf-8")

        manifest.append(
            {
                "record_id": record_id,
                "pdf_path": str(pdf_path),
                "xml_path": str(xml_path),
            }
        )

    return manifest


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def collect_text(element: ET.Element) -> str:
    return " ".join(part.strip() for part in element.itertext() if part and part.strip())


def read_tei_header(tei_path: Path, max_chars: int = 12000) -> str:
    text = tei_path.read_text(encoding="utf-8", errors="ignore")
    start = text.find("<teiHeader")
    end = text.find("</teiHeader>")
    if start == -1 or end == -1:
        return text[:max_chars]
    return text[start : end + len("</teiHeader>")][:max_chars]


def extract_alto_lines(alto_path: Path) -> List[LayoutLine]:
    tree = ET.parse(alto_path)
    root = tree.getroot()
    lines: List[LayoutLine] = []
    page_index = 0
    for page in root.iter():
        if strip_ns(page.tag) != "Page":
            continue
        for line in page.iter():
            if strip_ns(line.tag) != "TextLine":
                continue
            parts = [
                child.attrib.get("CONTENT", "").strip()
                for child in list(line)
                if strip_ns(child.tag) == "String" and child.attrib.get("CONTENT", "").strip()
            ]
            if not parts:
                continue
            lines.append(
                {
                    "text": " ".join(parts),
                    "x": float(line.attrib.get("HPOS", "0") or 0),
                    "y": float(line.attrib.get("VPOS", "0") or 0),
                    "page": page_index,
                }
            )
        page_index += 1
    lines.sort(key=lambda item: (item["page"], item["y"], item["x"]))
    return lines


def extract_oai_dc(xml_path: Path) -> MetadataRecord:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    namespaces = {"dc": "http://purl.org/dc/elements/1.1/"}

    def values(tag: str) -> List[str]:
        result = []
        for element in root.findall(f".//dc:{tag}", namespaces):
            if element.text and element.text.strip():
                result.append(element.text.strip())
        return result

    def values_with_language(tag: str) -> List[Tuple[str, str]]:
        result: List[Tuple[str, str]] = []
        for element in root.findall(f".//dc:{tag}", namespaces):
            if not element.text or not element.text.strip():
                continue
            language = (
                element.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") or element.attrib.get("lang") or ""
            )
            result.append((language, element.text.strip()))
        return result

    titles = values("title")
    descriptions = values("description")
    subject_entries = values_with_language("subject")
    keywords = [text for _, text in subject_entries]
    keyword_groups: Dict[str, List[str]] = {}
    for language, text in subject_entries:
        keyword_groups.setdefault(language or "unknown", []).append(text)

    return {
        "title": (titles or [""])[0],
        "titles": titles,
        "authors": values("creator"),
        "abstract": (descriptions or [""])[0],
        "abstracts": descriptions,
        "keywords": keywords,
        "keywords_groups": keyword_groups,
        "publisher": (values("publisher") or [""])[0],
        "date": (values("date") or [""])[0],
        "language": (values("language") or [""])[0],
        "identifiers": values("identifier"),
        "relations": values("relation"),
        "rights": (values("rights") or [""])[0],
        "types": values("type"),
        "formats": values("format"),
    }


def extract_tei_fields(tei_path: Path) -> MetadataRecord:
    tree = ET.parse(tei_path)
    root = tree.getroot()

    title = ""
    for element in root.iter():
        if strip_ns(element.tag) == "title":
            if element.attrib.get("type") == "main" or not title:
                title = collect_text(element)
                if title:
                    break

    authors: List[str] = []
    for element in root.iter():
        if strip_ns(element.tag) == "persName":
            name = collect_text(element)
            if name:
                authors.append(name)

    abstract = ""
    for element in root.iter():
        if strip_ns(element.tag) == "abstract":
            abstract = collect_text(element)
            if abstract:
                break

    keywords: List[str] = []
    for element in root.iter():
        if strip_ns(element.tag) == "term":
            term = collect_text(element)
            if term:
                keywords.append(term)

    # Collect <idno> only from the article's own header (skip <listBibl>, which
    # holds cited references). Filter by the type attribute to keep publication
    # IDs (DOI/PMID/PMCID/arXiv/ISSN/ISBN/URL); drop Grobid-internal types like
    # MD5 hashes and grant numbers.
    _IDNO_KEEP_TYPES = {"doi", "pmid", "pmcid", "arxiv", "issn", "isbn", "url", ""}
    identifiers: List[str] = []

    def _walk_for_idno(node: ET.Element) -> None:
        for child in list(node):
            tag = strip_ns(child.tag)
            if tag == "listBibl":
                continue
            if tag == "idno":
                idno_type = (child.attrib.get("type") or "").lower()
                if idno_type not in _IDNO_KEEP_TYPES:
                    continue
                value = collect_text(child)
                if value:
                    identifiers.append(value)
            else:
                _walk_for_idno(child)

    file_desc = None
    for element in root.iter():
        if strip_ns(element.tag) == "fileDesc":
            file_desc = element
            break
    if file_desc is not None:
        _walk_for_idno(file_desc)
    else:
        _walk_for_idno(root)

    language = ""
    for element in root.iter():
        if strip_ns(element.tag) == "teiHeader":
            language = element.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "")
            break

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "keywords": keywords,
        "identifiers": identifiers,
        "language": language,
    }


def extract_tei_abstracts(tei_path: Path) -> List[str]:
    try:
        tree = ET.parse(tei_path)
    except Exception:
        return []
    root = tree.getroot()
    abstracts: List[str] = []
    for element in root.iter():
        if strip_ns(element.tag) == "abstract":
            text = collect_text(element)
            if text:
                abstracts.append(text)
    return abstracts


def extract_jats_fields(xml_path: Path) -> MetadataRecord:
    """Parse a JATS XML article into a metadata dict with both the 13 DC-shaped
    header fields and 6 content fields (body_sections, figure_captions,
    table_captions, reference_dois, reference_titles, reference_records,
    formula_count).

    Header fields are scoped to the first <article-meta> element so references
    inside <back>/<ref-list> do not leak into the paper's own title/authors/
    keywords.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    article_meta = None
    for el in root.iter():
        if strip_ns(el.tag) == "article-meta":
            article_meta = el
            break
    scope = article_meta if article_meta is not None else root

    titles: List[str] = []
    for el in scope.iter():
        if strip_ns(el.tag) in {"article-title", "trans-title"}:
            t = collect_text(el)
            if t:
                titles.append(t)
    title = titles[0] if titles else ""

    authors: List[str] = []
    for contrib in scope.iter():
        if strip_ns(contrib.tag) != "contrib":
            continue
        ctype = contrib.get("contrib-type", "author")
        if ctype and ctype.lower() != "author":
            continue
        for name_el in contrib.iter():
            if strip_ns(name_el.tag) != "name":
                continue
            surname = given = ""
            for child in name_el:
                tag = strip_ns(child.tag)
                if tag == "surname":
                    surname = collect_text(child)
                elif tag == "given-names":
                    given = collect_text(child)
            if surname:
                authors.append(f"{given} {surname}".strip())
            break

    abstracts: List[str] = []
    for el in scope.iter():
        if strip_ns(el.tag) in {"abstract", "trans-abstract"}:
            t = collect_text(el)
            if t:
                abstracts.append(t)

    keywords: List[str] = []
    keyword_groups: Dict[str, List[str]] = {}
    for kg in scope.iter():
        if strip_ns(kg.tag) != "kwd-group":
            continue
        lang_key = (
            kg.get("{http://www.w3.org/XML/1998/namespace}lang")
            or kg.get("lang")
            or "unknown"
        )
        group: List[str] = []
        for el in kg.iter():
            if strip_ns(el.tag) == "kwd":
                t = collect_text(el)
                if t:
                    group.append(t)
                    keywords.append(t)
        if group:
            keyword_groups.setdefault(lang_key, []).extend(group)
    if not keywords:
        for el in scope.iter():
            if strip_ns(el.tag) == "kwd":
                t = collect_text(el)
                if t:
                    keywords.append(t)

    doi = ""
    for el in scope.iter():
        if strip_ns(el.tag) == "article-id" and el.get("pub-id-type") == "doi":
            doi = collect_text(el)
            break

    language = root.get("{http://www.w3.org/XML/1998/namespace}lang", "")

    publisher = ""
    for el in scope.iter():
        if strip_ns(el.tag) == "publisher-name":
            publisher = collect_text(el)
            break

    body_sections: List[str] = []
    body_el = None
    for el in root.iter():
        if strip_ns(el.tag) == "body":
            body_el = el
            break
    if body_el is not None:
        for sec in body_el.iter():
            if strip_ns(sec.tag) != "sec":
                continue
            for child in sec:
                if strip_ns(child.tag) == "title":
                    t = collect_text(child)
                    if t:
                        body_sections.append(t)
                    break

    figure_captions: List[str] = []
    for fig in root.iter():
        if strip_ns(fig.tag) != "fig":
            continue
        label_text = ""
        caption_text = ""
        for child in fig:
            tag = strip_ns(child.tag)
            if tag == "label":
                label_text = collect_text(child)
            elif tag == "caption":
                caption_text = collect_text(child)
        combined = " ".join(part for part in (label_text, caption_text) if part).strip()
        if combined:
            figure_captions.append(combined)

    table_captions: List[str] = []
    for tw in root.iter():
        if strip_ns(tw.tag) != "table-wrap":
            continue
        label_text = ""
        caption_text = ""
        for child in tw:
            tag = strip_ns(child.tag)
            if tag == "label":
                label_text = collect_text(child)
            elif tag == "caption":
                caption_text = collect_text(child)
        combined = " ".join(part for part in (label_text, caption_text) if part).strip()
        if combined:
            table_captions.append(combined)

    reference_dois: List[str] = []
    reference_titles: List[str] = []
    reference_records: List[Dict[str, str]] = []
    for ref in root.iter():
        if strip_ns(ref.tag) != "ref":
            continue
        got_doi = ""
        for inner in ref.iter():
            if strip_ns(inner.tag) == "pub-id" and (inner.get("pub-id-type") or "").lower() == "doi":
                got_doi = collect_text(inner)
                if got_doi:
                    break
        if got_doi:
            reference_dois.append(got_doi)
        got_title = ""
        for inner in ref.iter():
            if strip_ns(inner.tag) == "article-title":
                got_title = collect_text(inner)
                if got_title:
                    break
        if not got_title:
            for inner in ref.iter():
                if strip_ns(inner.tag) == "source":
                    got_title = collect_text(inner)
                    if got_title:
                        break
        if got_title:
            reference_titles.append(got_title)
        if got_doi or got_title:
            reference_records.append({"doi": got_doi, "title": got_title})

    formula_count = sum(
        1 for el in root.iter() if strip_ns(el.tag) in {"disp-formula", "inline-formula"}
    )

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
        "language": language,
        "identifiers": [doi] if doi else [],
        "relations": [],
        "rights": "",
        "types": [],
        "formats": [],
        "body_sections": body_sections,
        "figure_captions": figure_captions,
        "table_captions": table_captions,
        "reference_dois": reference_dois,
        "reference_titles": reference_titles,
        "reference_records": reference_records,
        "formula_count": formula_count,
    }


def extract_tei_content_fields(tei_path: Path) -> MetadataRecord:
    """Extract body/figure/table/reference content from a GROBID TEI file.

    Returns the six content-field keys that complement extract_tei_fields' header
    output. Shape is comparable to extract_jats_fields' content-side keys so
    downstream merging is straightforward.
    """
    try:
        tree = ET.parse(tei_path)
    except Exception:
        return {
            "body_sections": [],
            "figure_captions": [],
            "table_captions": [],
            "reference_dois": [],
            "reference_titles": [],
            "formula_count": 0,
        }
    root = tree.getroot()

    body_sections: List[str] = []
    body_el = None
    for el in root.iter():
        if strip_ns(el.tag) == "body":
            body_el = el
            break
    if body_el is not None:
        for div in body_el.iter():
            if strip_ns(div.tag) != "div":
                continue
            for child in div:
                if strip_ns(child.tag) == "head":
                    t = collect_text(child)
                    if t:
                        body_sections.append(t)
                    break

    figure_captions: List[str] = []
    table_captions: List[str] = []
    for fig in root.iter():
        if strip_ns(fig.tag) != "figure":
            continue
        kind = (fig.get("type") or "").lower()
        head_text = ""
        desc_text = ""
        for child in fig:
            tag = strip_ns(child.tag)
            if tag == "head":
                head_text = collect_text(child)
            elif tag == "figDesc":
                desc_text = collect_text(child)
        combined = " ".join(p for p in (head_text, desc_text) if p).strip()
        if not combined:
            continue
        if kind == "table":
            table_captions.append(combined)
        else:
            figure_captions.append(combined)

    reference_dois: List[str] = []
    reference_titles: List[str] = []
    doi_re = re.compile(r"10\.\d{4,9}/[^\s<>\"'\\]+", re.IGNORECASE)
    for bibl in root.iter():
        if strip_ns(bibl.tag) != "biblStruct":
            continue
        got_doi = ""
        for inner in bibl.iter():
            tag = strip_ns(inner.tag)
            if tag == "idno" and (inner.get("type") or "").lower() == "doi":
                got_doi = collect_text(inner)
                if got_doi:
                    break
        if not got_doi:
            for inner in bibl.iter():
                if strip_ns(inner.tag) == "ptr":
                    target = inner.get("target") or ""
                    m = doi_re.search(target)
                    if m:
                        got_doi = m.group(0)
                        break
        if got_doi:
            reference_dois.append(got_doi)
        got_title = ""
        for inner in bibl.iter():
            if strip_ns(inner.tag) != "title":
                continue
            level = (inner.get("level") or "").lower()
            if level == "a":
                got_title = collect_text(inner)
                if got_title:
                    break
        if got_title:
            reference_titles.append(got_title)

    formula_count = sum(1 for el in root.iter() if strip_ns(el.tag) == "formula")

    return {
        "body_sections": body_sections,
        "figure_captions": figure_captions,
        "table_captions": table_captions,
        "reference_dois": reference_dois,
        "reference_titles": reference_titles,
        "formula_count": formula_count,
    }


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return cast(Dict[str, Any], json.loads(text))
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in response")


def safe_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def normalize_metadata(record: MetadataRecord) -> MetadataRecord:
    return {
        "title": str(record.get("title", "")).strip(),
        "authors": safe_list(record.get("authors")),
        "affiliations": safe_list(record.get("affiliations")),
        "abstract": str(record.get("abstract", "")).strip(),
        "keywords": safe_list(record.get("keywords")),
        "publisher": str(record.get("publisher", "")).strip(),
        "date": str(record.get("date", "")).strip(),
        "language": str(record.get("language", "")).strip(),
        "identifiers": safe_list(record.get("identifiers")),
        "relations": safe_list(record.get("relations")),
        "rights": str(record.get("rights", "")).strip(),
        "types": safe_list(record.get("types")),
        "formats": safe_list(record.get("formats")),
    }
