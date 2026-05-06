# pylint: disable=too-many-lines
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


TEI_ABSTRACT_HEAD_RE = re.compile(r"^\s*(abstract|resumo|resumen)\s*:?\s*$", re.IGNORECASE)
TEI_ABSTRACT_HEAD_PREFIX_RE = re.compile(r"^\s*(abstract|resumo|resumen)\b\s*[:.-]?\s*(.*)$", re.IGNORECASE)
TEI_ABSTRACT_CONTINUATION_HEAD_RE = re.compile(
    r"^\s*(conclusion|conclusions|conclusao|conclusoes|conclusion:)\s*:?\s*$",
    re.IGNORECASE,
)
TEI_ABSTRACT_STOP_HEAD_RE = re.compile(
    r"^\s*(introduction|introducao|introduccion|methods?|metodos|metodologia|results?|resultados|"
    r"discussion|discussao|discusion|references?|referencias)\b",
    re.IGNORECASE,
)
TEI_STRUCTURED_ABSTRACT_START_RE = re.compile(
    r"^\s*(objective|objetivo|background|method|methods|metodo|metodos|results?|resultados|"
    r"introduction|introducao|introduccion)\s*[:.]",
    re.IGNORECASE,
)
TEI_STRUCTURED_ABSTRACT_HEAD_RE = re.compile(
    r"^\s*(summary|objective|objetivo|background|introduction|introducao|introduccion|"
    r"method|methods|metodologia|metodología|metodo|metodos|"
    r"results?|resultados|conclusions?|conclusiones|conclusoes|conclusões)\b",
    re.IGNORECASE,
)
TEI_STRUCTURED_ABSTRACT_START_HEAD_RE = re.compile(
    r"^\s*(summary|objective|objetivo|background|introduction|introducao|introduccion)\b",
    re.IGNORECASE,
)
TEI_KEYWORD_LABEL_RE = re.compile(
    r"\b(keywords?|key-words?|palavras[-\s]+chave|palabras\s+claves?|descritores?|descriptors?)\s*[:.]?\s*",
    re.IGNORECASE,
)
TEI_DISCLOSURE_RE = re.compile(
    r"\b(conflitos?\s+de\s+interesse|conflicts?\s+of\s+interest|fontes?\s+de\s+financiamento|"
    r"funding|declaram|declare)\b",
    re.IGNORECASE,
)
TEI_INLINE_SUMMARY_RE = re.compile(r"\b(summary|abstract)\b\s*[:.]?\s+(.+)$", re.IGNORECASE)
TEI_LEAD_BLEED_RE = re.compile(
    r"\s+(?:conforme\s+demonstrado\s+na\s+figura\b|as\s+shown\s+in\s+fig(?:ure)?\b|"
    r"a\s+seguir\b|below\s+we\s+describe\b).*$",
    re.IGNORECASE,
)


def _direct_head_text(element: ET.Element) -> str:
    for child in list(element):
        if strip_ns(child.tag) == "head":
            return collect_text(child)
    return ""


def _abstract_head_tail(head: str) -> str:
    match = TEI_ABSTRACT_HEAD_PREFIX_RE.match(head or "")
    if not match:
        return ""
    return match.group(2).strip(" ;.:-")


def _split_keyword_tail(text: str) -> Tuple[str, List[str]]:
    text = re.sub(r"\s+", " ", text or "").strip()
    match = TEI_KEYWORD_LABEL_RE.search(text or "")
    if not match:
        return text, []
    before = text[: match.start()].strip(" ;:-")
    tail = text[match.end() :].strip()
    summary_match = TEI_INLINE_SUMMARY_RE.search(tail)
    if summary_match:
        summary_text = tail[summary_match.start() :].strip(" ;.:-")
        before = " ".join(part for part in (before, summary_text) if part).strip()
        tail = tail[: summary_match.start()].strip()
    terms = [part.strip(" ;.:-") for part in re.split(r"\s*[;,]\s*|\.\s+", tail) if part.strip(" ;.:-")]
    return before, terms


def _inline_summary_tail(text: str) -> str:
    match = TEI_INLINE_SUMMARY_RE.search(text or "")
    if not match:
        return ""
    tail = match.group(2).strip()
    return tail if len(re.findall(r"\w+", tail)) >= 20 else ""


def _trim_body_lead_bleed(text: str) -> str:
    trimmed, count = TEI_LEAD_BLEED_RE.subn("", text or "", count=1)
    if not count:
        return text
    trimmed = trimmed.strip(" ;,.:-")
    if len(re.findall(r"\w+", trimmed)) >= 40:
        return trimmed
    return text


def _abstract_div_text_and_keywords(element: ET.Element) -> Tuple[str, List[str]]:
    parts: List[str] = []
    keywords: List[str] = []
    for child in list(element):
        tag = strip_ns(child.tag)
        if tag == "head":
            continue
        if tag == "p":
            text, terms = _split_keyword_tail(collect_text(child))
            if text:
                parts.append(text)
            keywords.extend(terms)
    if not parts:
        text, terms = _split_keyword_tail(collect_text(element))
        head = _direct_head_text(element)
        if head and text.startswith(head):
            text = text[len(head) :].strip(" ;.:-")
        if text:
            parts.append(text)
        keywords.extend(terms)
    return " ".join(parts).strip(), keywords


def _abstract_chunks_from_container(container: ET.Element) -> Tuple[List[str], List[str]]:
    divs = [child for child in list(container) if strip_ns(child.tag) == "div"]
    keywords: List[str] = []
    if not divs:
        text, terms = _split_keyword_tail(collect_text(container))
        return ([text] if text else []), terms

    language_headed = sum(
        1
        for d in divs
        if TEI_ABSTRACT_HEAD_RE.match(_direct_head_text(d) or "")
        or _abstract_head_tail(_direct_head_text(d) or "")
    )
    if language_headed == 0:
        parts: List[str] = []
        for d in divs:
            body, terms = _abstract_div_text_and_keywords(d)
            if body:
                parts.append(body)
            keywords.extend(terms)
        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        return ([text] if text else []), keywords

    chunks: List[str] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        text = " ".join(part for part in current if part).strip()
        if text:
            chunks.append(text)
        current = []

    for div in divs:
        head = _direct_head_text(div)
        body, terms = _abstract_div_text_and_keywords(div)
        keywords.extend(terms)
        head_tail = _abstract_head_tail(head)
        head_is_abstract = bool(TEI_ABSTRACT_HEAD_RE.match(head)) or bool(head_tail)
        head_is_continuation = bool(TEI_ABSTRACT_CONTINUATION_HEAD_RE.match(head))
        head_is_stop = bool(TEI_ABSTRACT_STOP_HEAD_RE.match(head))
        starts_like_abstract = bool(TEI_STRUCTURED_ABSTRACT_START_RE.match(body))

        if head_is_stop:
            flush()
            break
        if head_is_abstract:
            flush()
            text = " ".join(part for part in (head_tail, body) if part).strip()
            current = [text] if text else []
            continue
        if head_is_continuation and current:
            if body:
                current.append(body)
            continue
        if starts_like_abstract:
            if current:
                current.append(body)
            else:
                current = [body]
            continue
        if current and not head:
            if body:
                current.append(body)
            continue
        flush()
    flush()
    return chunks, keywords


def _extract_body_lead_abstract_candidates(root: ET.Element) -> List[str]:
    body = None
    for element in root.iter():
        if strip_ns(element.tag) == "body":
            body = element
            break
    if body is None:
        return []

    candidates: List[str] = []
    for div in body.iter():
        if strip_ns(div.tag) != "div":
            continue
        head = _direct_head_text(div)
        if TEI_ABSTRACT_STOP_HEAD_RE.match(head) or TEI_DISCLOSURE_RE.search(head):
            continue
        paragraphs: List[str] = []
        for child in list(div):
            if strip_ns(child.tag) != "p":
                continue
            text, _ = _split_keyword_tail(collect_text(child))
            if text:
                paragraphs.append(text)
            current_words = len(re.findall(r"\w+", " ".join(paragraphs)))
            if current_words >= 40 or len(paragraphs) >= 2:
                break
        text = " ".join(paragraphs).strip()
        text = _trim_body_lead_bleed(text)
        word_count = len(re.findall(r"\w+", text))
        if 40 <= word_count <= 350 and not TEI_DISCLOSURE_RE.search(text):
            candidates.append(text)
            break
    return candidates


def _extract_body_structured_abstract_candidates(root: ET.Element) -> List[str]:
    body = None
    for element in root.iter():
        if strip_ns(element.tag) == "body":
            body = element
            break
    if body is None:
        return []

    candidates: List[str] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        text = " ".join(part for part in current if part).strip()
        word_count = len(re.findall(r"\w+", text))
        if 40 <= word_count <= 700 and not TEI_DISCLOSURE_RE.search(text):
            candidates.append(text)
        current = []

    for div in body.iter():
        if strip_ns(div.tag) != "div":
            continue
        head = _direct_head_text(div)
        head_is_structured = bool(TEI_STRUCTURED_ABSTRACT_HEAD_RE.match(head or ""))
        head_starts_sequence = bool(TEI_STRUCTURED_ABSTRACT_START_HEAD_RE.match(head or ""))
        if not head_is_structured:
            if current:
                flush()
            if candidates:
                break
            continue
        body_text, _ = _abstract_div_text_and_keywords(div)
        text = " ".join(part for part in (head.rstrip(" :."), body_text) if part).strip()
        if not text:
            continue
        if current and head_starts_sequence:
            carry = _inline_summary_tail(" ".join(current))
            flush()
            if carry:
                current.append(carry)
        current.append(text)
    if current:
        flush()
    return candidates


def _extract_tei_abstracts_and_keywords_from_root(root: ET.Element) -> Tuple[List[str], List[str]]:
    abstracts: List[str] = []
    keywords: List[str] = []
    seen = set()

    def add_text(text: str) -> None:
        value = re.sub(r"\s+", " ", text or "").strip()
        key = value.casefold()
        if value and key not in seen:
            abstracts.append(value)
            seen.add(key)

    abstract_has_raw_content = False
    for element in root.iter():
        if strip_ns(element.tag) != "abstract":
            continue
        if collect_text(element).strip():
            abstract_has_raw_content = True
        chunks, terms = _abstract_chunks_from_container(element)
        for chunk in chunks:
            add_text(chunk)
        keywords.extend(terms)

    for element in root.iter():
        if strip_ns(element.tag) != "div":
            continue
        head = _direct_head_text(element)
        head_tail = _abstract_head_tail(head)
        if not TEI_ABSTRACT_HEAD_RE.match(head) and not head_tail:
            continue
        text, terms = _abstract_div_text_and_keywords(element)
        if head_tail:
            text = " ".join(part for part in (head_tail, text) if part).strip()
        add_text(text)
        keywords.extend(terms)

    if not abstract_has_raw_content:
        for text in _extract_body_structured_abstract_candidates(root):
            add_text(text)
        if not abstracts:
            for text in _extract_body_lead_abstract_candidates(root):
                add_text(text)

    deduped_keywords: List[str] = []
    seen_keywords = set()
    for keyword in keywords:
        key = re.sub(r"\s+", " ", keyword or "").strip().casefold()
        if key and key not in seen_keywords:
            deduped_keywords.append(keyword)
            seen_keywords.add(key)
    return abstracts, deduped_keywords


def read_tei_header(tei_path: Path, max_chars: int = 12000) -> str:
    text = tei_path.read_text(encoding="utf-8", errors="ignore")
    start = text.find("<teiHeader")
    end = text.find("</teiHeader>")
    if start == -1 or end == -1:
        return text[:max_chars]
    return text[start : end + len("</teiHeader>")][:max_chars]


def extract_alto_lines(alto_path: Path) -> List[LayoutLine]:
    from collections import Counter

    tree = ET.parse(alto_path)
    root = tree.getroot()
    styles: Dict[str, Dict[str, Any]] = {}
    for element in root.iter():
        if strip_ns(element.tag) != "TextStyle":
            continue
        style_id = element.attrib.get("ID")
        if not style_id:
            continue
        try:
            font_size = float(element.attrib.get("FONTSIZE", "0") or 0)
        except ValueError:
            font_size = 0.0
        font_style_raw = element.attrib.get("FONTSTYLE", "") or ""
        styles[style_id] = {
            "font_size": font_size,
            "font_style": font_style_raw,
            "is_superscript": "superscript" in font_style_raw.lower(),
            "is_bold": "bold" in font_style_raw.lower(),
            "is_italic": "italic" in font_style_raw.lower(),
            "font_family": (element.attrib.get("FONTFAMILY", "") or "").strip().lower(),
            "font_type": (element.attrib.get("FONTTYPE", "") or "").strip().lower(),
            "font_color": (element.attrib.get("FONTCOLOR", "") or "").strip().lower(),
        }

    lines: List[LayoutLine] = []
    page_index = 0
    for page in root.iter():
        if strip_ns(page.tag) != "Page":
            continue
        try:
            page_w = float(page.attrib.get("WIDTH", "0") or 0)
        except ValueError:
            page_w = 0.0
        try:
            page_h = float(page.attrib.get("HEIGHT", "0") or 0)
        except ValueError:
            page_h = 0.0
        col_pivot = page_w / 2.0 if page_w > 0 else 306.0
        block_index = 0
        for block in page.iter():
            if strip_ns(block.tag) != "TextBlock":
                continue
            try:
                block_hpos = float(block.attrib.get("HPOS", "0") or 0)
                block_vpos = float(block.attrib.get("VPOS", "0") or 0)
                block_w = float(block.attrib.get("WIDTH", "0") or 0)
                block_h = float(block.attrib.get("HEIGHT", "0") or 0)
            except ValueError:
                block_hpos = block_vpos = block_w = block_h = 0.0
            block_id = block.attrib.get("ID", f"p{page_index}_b{block_index}")
            block_text_lines = [c for c in block.iter() if strip_ns(c.tag) == "TextLine"]
            block_line_count = len(block_text_lines)
            for line_index_in_block, line in enumerate(block_text_lines):
                string_records: List[Dict[str, Any]] = []
                for child in list(line):
                    if strip_ns(child.tag) != "String":
                        continue
                    content = (child.attrib.get("CONTENT", "") or "").strip()
                    if not content:
                        continue
                    refs = (child.attrib.get("STYLEREFS", "") or "").split()
                    merged: Dict[str, Any] = {"content": content}
                    for ref in refs:
                        s = styles.get(ref, {})
                        if s.get("font_size") and not merged.get("font_size"):
                            merged["font_size"] = s["font_size"]
                        if s.get("is_superscript"):
                            merged["is_superscript"] = True
                        if s.get("is_bold"):
                            merged["is_bold"] = True
                        if s.get("is_italic"):
                            merged["is_italic"] = True
                        if s.get("font_family") and not merged.get("font_family"):
                            merged["font_family"] = s["font_family"]
                        if s.get("font_type") and not merged.get("font_type"):
                            merged["font_type"] = s["font_type"]
                        if s.get("font_color") and not merged.get("font_color"):
                            merged["font_color"] = s["font_color"]
                    string_records.append(merged)
                if not string_records:
                    continue

                non_super = [s for s in string_records if not s.get("is_superscript")]
                text_records = non_super if non_super else string_records
                text = " ".join(s["content"] for s in text_records)

                char_weights = [(len(s["content"]), s) for s in text_records]
                total_chars = sum(w for w, _ in char_weights) or 1
                size_buckets: "Counter[float]" = Counter()
                family_buckets: "Counter[str]" = Counter()
                type_buckets: "Counter[str]" = Counter()
                color_buckets: "Counter[str]" = Counter()
                bold_chars = 0
                italic_chars = 0
                for w, s in char_weights:
                    size = float(s.get("font_size") or 0.0)
                    if size > 0:
                        size_buckets[round(size * 10) / 10] += w
                    if s.get("font_family"):
                        family_buckets[s["font_family"]] += w
                    if s.get("font_type"):
                        type_buckets[s["font_type"]] += w
                    if s.get("font_color"):
                        color_buckets[s["font_color"]] += w
                    if s.get("is_bold"):
                        bold_chars += w
                    if s.get("is_italic"):
                        italic_chars += w
                modal_font_size = size_buckets.most_common(1)[0][0] if size_buckets else 0.0
                modal_bold = bold_chars * 2 > total_chars
                modal_italic = italic_chars * 2 > total_chars
                modal_family = family_buckets.most_common(1)[0][0] if family_buckets else ""
                modal_type = type_buckets.most_common(1)[0][0] if type_buckets else ""
                modal_color = color_buckets.most_common(1)[0][0] if color_buckets else ""

                style_refs_all: List[str] = []
                for child in list(line):
                    if strip_ns(child.tag) == "String":
                        for ref in (child.attrib.get("STYLEREFS", "") or "").split():
                            if ref:
                                style_refs_all.append(ref)

                lines.append(
                    {
                        "text": text,
                        "x": float(line.attrib.get("HPOS", "0") or 0),
                        "y": float(line.attrib.get("VPOS", "0") or 0),
                        "w": float(line.attrib.get("WIDTH", "0") or 0),
                        "h": float(line.attrib.get("HEIGHT", "0") or 0),
                        "font_size": modal_font_size,
                        "bold": modal_bold,
                        "italic": modal_italic,
                        "font_family": modal_family,
                        "font_type": modal_type,
                        "font_color": modal_color,
                        "style_refs": sorted(set(style_refs_all)),
                        "page": page_index,
                        "page_w": page_w,
                        "page_h": page_h,
                        "block_id": block_id,
                        "block_hpos": block_hpos,
                        "block_vpos": block_vpos,
                        "block_w": block_w,
                        "block_h": block_h,
                        "block_line_count": block_line_count,
                        "is_block_first_line": line_index_in_block == 0,
                        "block_col": 1 if block_hpos >= col_pivot else 0,
                        "strings": string_records,
                    }
                )
            block_index += 1
        page_index += 1
    lines.sort(
        key=lambda item: (
            item["page"],
            item.get("block_col", 0),
            item.get("block_vpos", item["y"]),
            item["y"],
            item["x"],
        )
    )

    if lines:
        body_family: "Counter[str]" = Counter()
        body_type: "Counter[str]" = Counter()
        body_color: "Counter[str]" = Counter()
        for ln in lines:
            txt = ln.get("text") or ""
            if len(txt) < 30 or ln.get("bold"):
                continue
            w = len(txt)
            if ln.get("font_family"):
                body_family[ln["font_family"]] += w
            if ln.get("font_type"):
                body_type[ln["font_type"]] += w
            if ln.get("font_color"):
                body_color[ln["font_color"]] += w
        doc_body_family = body_family.most_common(1)[0][0] if body_family else ""
        doc_body_type = body_type.most_common(1)[0][0] if body_type else ""
        doc_body_color = body_color.most_common(1)[0][0] if body_color else ""
        for ln in lines:
            ln["doc_body_family"] = doc_body_family
            ln["doc_body_type"] = doc_body_type
            ln["doc_body_color"] = doc_body_color
    return lines


def extract_oai_dc(xml_path: Path) -> MetadataRecord:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    namespaces = {"dc": "http://purl.org/dc/elements/1.1/"}

    def is_placeholder_value(text: str) -> bool:
        return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower()) in {"na", "nada", "nan", "none", "null"}

    def values(tag: str) -> List[str]:
        result = []
        for element in root.findall(f".//dc:{tag}", namespaces):
            if element.text and element.text.strip() and not is_placeholder_value(element.text):
                result.append(element.text.strip())
        return result

    def values_with_language(tag: str) -> List[Tuple[str, str]]:
        result: List[Tuple[str, str]] = []
        for element in root.findall(f".//dc:{tag}", namespaces):
            if not element.text or not element.text.strip() or is_placeholder_value(element.text):
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
    """Return raw GROBID TEI header fields. Each value is collect_text on the
    canonical element inside `<teiHeader>` with whitespace collapsed and no
    further chunking, splitting, type-filtering, or recovery heuristics.
    Cited references in `<text>/<back>/<listBibl>` are excluded by scoping
    to `<teiHeader>`.
    """
    tree = ET.parse(tei_path)
    root = tree.getroot()
    header = next((el for el in root.iter() if strip_ns(el.tag) == "teiHeader"), root)

    title = ""
    for element in header.iter():
        if strip_ns(element.tag) == "title":
            if element.attrib.get("type") == "main" or not title:
                title = collect_text(element)
                if title:
                    break

    authors: List[str] = []
    for author_el in header.iter():
        if strip_ns(author_el.tag) != "author":
            continue
        for element in author_el.iter():
            if strip_ns(element.tag) == "persName":
                name = collect_text(element)
                if name:
                    authors.append(name)

    abstract = ""
    for element in header.iter():
        if strip_ns(element.tag) != "abstract":
            continue
        raw = re.sub(r"\s+", " ", collect_text(element)).strip()
        if raw and not TEI_DISCLOSURE_RE.search(raw):
            abstract = raw
            break
    if not abstract:
        abstracts, _ = _extract_tei_abstracts_and_keywords_from_root(root)
        abstract = next(
            (value for value in abstracts if not TEI_DISCLOSURE_RE.search(value)),
            abstracts[0] if abstracts else "",
        )

    keywords: List[str] = []
    for element in header.iter():
        if strip_ns(element.tag) == "term":
            term = collect_text(element)
            if term:
                keywords.append(term)

    identifiers: List[str] = []
    for element in header.iter():
        if strip_ns(element.tag) == "idno":
            value = collect_text(element)
            if value:
                identifiers.append(value)

    language = header.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "")

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
    abstracts, _ = _extract_tei_abstracts_and_keywords_from_root(tree.getroot())
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
        if (fig.get("type") or "").lower() == "table":
            table_captions.append(combined)
        else:
            figure_captions.append(combined)

    reference_dois: List[str] = []
    reference_titles: List[str] = []
    for bibl in root.iter():
        if strip_ns(bibl.tag) != "biblStruct":
            continue
        for inner in bibl.iter():
            tag = strip_ns(inner.tag)
            if tag == "idno" and (inner.get("type") or "").lower() == "doi":
                value = collect_text(inner)
                if value:
                    reference_dois.append(value)
                    break
        for inner in bibl.iter():
            if strip_ns(inner.tag) != "title":
                continue
            if (inner.get("level") or "").lower() == "a":
                value = collect_text(inner)
                if value:
                    reference_titles.append(value)
                    break

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
