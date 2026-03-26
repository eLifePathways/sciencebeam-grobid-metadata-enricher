from __future__ import annotations

import csv
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
                element.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
                or element.attrib.get("lang")
                or ""
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

    identifiers: List[str] = []
    for element in root.iter():
        if strip_ns(element.tag) == "idno":
            identifier = collect_text(element)
            if identifier:
                identifiers.append(identifier)

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


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
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
