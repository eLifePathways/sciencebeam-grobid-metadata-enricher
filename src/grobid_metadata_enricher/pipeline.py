# pylint: disable=too-many-lines
from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .clients import (
    DEFAULT_GROBID_URL,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PARSER,
    DEFAULT_PDFALTO_BIN,
    DEFAULT_POOL_PATH,
    AoaiPool,
    OpenAIClient,
    run_grobid,
    run_pdfalto,
)
from .evaluation import aggregate_metrics, evaluate_record, write_root_cause_report
from .formats import (
    LayoutLine,
    ManifestRow,
    MetadataRecord,
    ensure_dir,
    extract_alto_lines,
    extract_json_from_text,
    extract_oai_dc,
    extract_tei_abstracts,
    extract_tei_fields,
    load_manifest,
    load_parquet_manifest,
    normalize_metadata,
    read_tei_header,
)
from .prompts import (
    ABSTRACT_EXTRACTION_PROMPT,
    ABSTRACT_SELECTION_PROMPT,
    BODY_SECTIONS_EXTRACTION_PROMPT,
    FIGURE_CAPTIONS_SELECTION_PROMPT,
    HEADER_METADATA_PROMPT,
    IDENTIFIER_SELECTION_PROMPT,
    KEYWORD_EXTRACTION_PROMPT,
    KEYWORD_INFERENCE_PROMPT,
    KEYWORD_SELECTION_PROMPT,
    KEYWORD_TRANSLATION_PROMPT,
    OCR_CLEANUP_PROMPT,
    REFERENCES_EXTRACTION_PROMPT,
    TABLE_CAPTIONS_SELECTION_PROMPT,
    TEI_METADATA_PROMPT,
)
from .telemetry import get_tracer, init_telemetry, with_otel_context

DISCLAIMER_RE = re.compile(
    r"(preprint|scielo|deposit|submitted|presentado|condi[cç]iones|condi[cç][aã]o|declaram|"
    r"responsab|conflicts?\s+of\s+interest|conflitos?\s+de\s+interesse|conflictos?\s+de\s+inter[eé]s|"
    r"potenciais\s+conflitos|funding\s*(?:statement|sources?|information)?\s*:|"
    r"financiamento\s*:|financiamiento\s*:|fuentes?\s+de\s+financiamiento|"
    r"fontes?\s+de\s+financiamento)",
    re.IGNORECASE,
)
ABSTRACT_MARKER_RE = re.compile(r"\b(abstract|resumo|resumen)\b", re.IGNORECASE)
# A line that LOOKS like a section heading "Abstract" / "Resumo" / "Resumen",
# possibly with a section number prefix or trailing punctuation. Lines longer
# than ~60 chars are paragraph text, not headings (filters out reference list
# entries containing "v1.abstract" or "Abstract 5463: …").
ABSTRACT_HEADING_RE = re.compile(
    r"^\s*(?:[\dIVXivx]+(?:\.\d+)*[\.\)]?\s+)?(abstract|resumo|resumen)\s*[\.:]?\s*$",
    re.IGNORECASE,
)
ENGLISH_MARKER_RE = re.compile(r"\babstract\b", re.IGNORECASE)
# Stop at front-matter/body boundaries. Structured abstract labels such as
# Background, Methods, Results, and Conclusion are intentionally excluded here:
# several publishers typeset them as standalone lines inside the abstract.
NEXT_SECTION_RE = re.compile(
    r"^\s*(?:[\dIVXivx]+(?:\.\d+)*[\.\)]?\s+)?"
    r"(?:introduction|introduccion|introdução|"
    r"discussion|discussão|discusión|"
    r"references|referencias|referências|"
    r"keywords?|palabras\s+clave|palavras-chave|"
    r"acknowledg|agradec|funding|financiamento|"
    r"highlights|significance|"
    r"author\s+summary|plain\s+language\s+summary|"
    r"abstract|resumo|resumen)"
    r"(?!\w)",
    re.IGNORECASE,
)
KEYWORD_MARKER_RE = re.compile(
    r"\b(keywords?|palavras[-\s]+chave|palabras\s+clave|descritores?|descriptors?)\b",
    re.IGNORECASE,
)
SUSPECT_KEYWORD_RE = re.compile(
    r"\b(conflicts?\s+of\s+interest|conflito\s+de\s+interesses?|funding|financiamento|"
    r"author'?s?\s+contributions?|author\s+roles?|plain\s+language\s+summary|"
    r"conceptualization|data\s+curation|formal\s+analysis|"
    r"investigation|resources|writing[-\s]+original\s+draft|writing[-\s]+review)\b",
    re.IGNORECASE,
)
PORTUGUESE_MARKER_RE = re.compile(r"\bresumo\b", re.IGNORECASE)
SPANISH_MARKER_RE = re.compile(r"\bresumen\b", re.IGNORECASE)
ENGLISH_START_RE = re.compile(r"^\s*(abstract|the|this|we|in|a|an)\b", re.IGNORECASE)
ROMANCE_START_RE = re.compile(r"^\s*(resumo|resumen)[:\s]", re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-ZáéíóúãõçñÁÉÍÓÚÃÕÇÑ]+")
TEXT_TOKEN_RE = re.compile(r"\w+")
LANGUAGE_STOPWORDS = {
    "en": {
        "the",
        "of",
        "and",
        "to",
        "in",
        "a",
        "is",
        "that",
        "for",
        "with",
        "as",
        "on",
        "are",
        "this",
        "was",
        "were",
        "be",
        "by",
        "it",
        "from",
        "or",
        "an",
        "which",
        "at",
    },
    "pt": {
        "o",
        "a",
        "e",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "um",
        "uma",
        "para",
        "por",
        "com",
        "como",
        "que",
        "se",
        "ao",
        "aos",
        "às",
    },
    "es": {
        "el",
        "la",
        "los",
        "las",
        "de",
        "y",
        "en",
        "a",
        "un",
        "una",
        "para",
        "por",
        "con",
        "como",
        "que",
        "se",
        "al",
        "del",
    },
}
SCIELO_RECORD_RE = re.compile(r"preprint_(\d+)$")
SCIELO_VIEW_URL = "https://preprints.scielo.org/index.php/scielo/preprint/view/{id}"
SCIELO_DOI = "10.1590/SciELOPreprints.{id}"
LIST_FIELDS = ("authors", "affiliations", "keywords", "identifiers", "relations", "types", "formats")
SCALAR_FIELDS = ("title", "abstract", "publisher", "date", "language", "rights")


@dataclass(frozen=True)
class PipelineSettings:
    manifest_path: Path
    pool_path: Path = DEFAULT_POOL_PATH
    openai_api_key: Optional[str] = DEFAULT_OPENAI_API_KEY
    openai_model: Optional[str] = DEFAULT_OPENAI_MODEL
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    output_dir: Path = Path("output")
    grobid_url: str = DEFAULT_GROBID_URL
    parser: str = DEFAULT_PARSER
    pdfalto_bin: Path = DEFAULT_PDFALTO_BIN
    pdfalto_start_page: int = 1
    pdfalto_end_page: int = 99
    pdfalto_header_end_page: int = 2
    limit: Optional[int] = None
    rerun: bool = False
    workers: int = 20
    per_document_llm_workers: int = 5
    llm_concurrency: int = 20
    llm_pool_routing: Optional[str] = None


@dataclass(frozen=True)
class DocumentPaths:
    record_id: str
    pdf_path: Path
    xml_path: Path
    tei_path: Path
    alto_path: Path
    prediction_path: Path


@dataclass(frozen=True)
class DocumentContext:
    record_id: str
    header_text: str
    lines: List[LayoutLine]
    first_page_lines: List[LayoutLine]
    tei_fields: MetadataRecord
    tei_abstracts: List[str]


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def _layout_line_page(line: LayoutLine) -> int:
    try:
        return int(line.get("page", 0) or 0)
    except (TypeError, ValueError):
        return 0


def front_matter_layout_lines(
    context: DocumentContext,
    *,
    max_lines: int,
    max_page: int = 5,
) -> List[LayoutLine]:
    lines = [line for line in context.lines if _layout_line_page(line) <= max_page]
    return (lines or context.first_page_lines or context.lines)[:max_lines]


_LAYOUT_FURNITURE_RE = re.compile(
    r"\b(?:bioRxiv|medRxiv)\s+preprint\b|"
    r"\b(?:this\s+version\s+posted|copyright\s+holder|not\s+certified\s+by\s+peer\s+review|"
    r"author/funder|license\s+to\s+display|made\s+available\s+under|cc[-\s]?by|"
    r"international\s+license|all\s+rights\s+reserved|downloaded\s+from)\b",
    re.IGNORECASE,
)
_LAYOUT_IDENTIFIER_RE = re.compile(r"\b(?:doi\s*:|https?://|www\.)", re.IGNORECASE)
_LAYOUT_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")


def _layout_text(line: LayoutLine) -> str:
    return normalize_whitespace(str(line.get("text", "") or ""))


def _layout_repeated_keys(lines: Sequence[LayoutLine], min_pages: int = 3) -> set[str]:
    pages_by_key: Dict[str, set[int]] = {}
    for line in lines:
        text = _layout_text(line)
        if not text:
            continue
        key = text.casefold()
        pages_by_key.setdefault(key, set()).add(_layout_line_page(line))
    return {key for key, pages in pages_by_key.items() if len(pages) >= min_pages}


def _page_y_scale(line: LayoutLine) -> float:
    page_h = float(line.get("page_h", 0.0) or 0.0)
    if page_h < 100.0:
        return 1.0
    return page_h / 792.0


def _page_x_scale(line: LayoutLine) -> float:
    page_w = float(line.get("page_w", 0.0) or 0.0)
    if page_w < 100.0:
        return 1.0
    return page_w / 612.0


def is_layout_furniture_line(
    line: LayoutLine,
    repeated_keys: set[str],
    *,
    drop_identifiers: bool = False,
) -> bool:
    text = _layout_text(line)
    if not text:
        return True
    if _LAYOUT_PUNCT_ONLY_RE.fullmatch(text):
        return True
    if re.fullmatch(r"[\divxIVX]+", text):
        return True
    if _LAYOUT_FURNITURE_RE.search(text):
        return True
    if re.fullmatch(r"doi\s*:?", text, re.IGNORECASE):
        return True
    if drop_identifiers and len(text) < 180 and _LAYOUT_IDENTIFIER_RE.search(text):
        return True

    y = float(line.get("y", 0.0) or 0.0)
    x = float(line.get("x", 0.0) or 0.0)
    ys = _page_y_scale(line)
    xs = _page_x_scale(line)
    key = text.casefold()
    if key in repeated_keys and (y < 65 * ys or y > 720 * ys or len(text) < 90):
        return True
    if x < 60 * xs and len(text) <= 6 and re.fullmatch(r"[\dA-Za-z.:-]+", text):
        return True
    if y < 32 * ys and len(text) < 180:
        return True
    if y > 760 * ys and len(text) < 100:
        return True
    return False


def prune_layout_lines(
    lines: Sequence[LayoutLine],
    *,
    max_page: Optional[int] = None,
    drop_identifiers: bool = False,
) -> List[LayoutLine]:
    repeated_keys = _layout_repeated_keys(lines)
    out: List[LayoutLine] = []
    for line in lines:
        if max_page is not None and _layout_line_page(line) > max_page:
            continue
        if is_layout_furniture_line(line, repeated_keys, drop_identifiers=drop_identifiers):
            continue
        out.append(line)
    return out


def front_matter_evidence_lines(
    context: DocumentContext,
    *,
    max_lines: int,
    max_page: int = 5,
    drop_identifiers: bool = False,
) -> List[LayoutLine]:
    source = [line for line in context.lines if _layout_line_page(line) <= max_page]
    source = source or context.first_page_lines or context.lines
    pruned = prune_layout_lines(source, drop_identifiers=drop_identifiers)
    return (pruned or source)[:max_lines]


def safe_extract_json(text: str) -> MetadataRecord:
    try:
        return extract_json_from_text(text)
    except Exception:
        return {}


def build_document_paths(
    row: ManifestRow,
    output_dir: Path,
    parser: str = DEFAULT_PARSER,
) -> DocumentPaths:
    # TEI and predictions are namespaced by parser so a second run against
    # the other backend does not silently re-use the first run's cached
    # outputs. ALTO is parser-independent (pdfalto on the raw PDF) so it
    # stays at the top level and is reused across parsers.
    return DocumentPaths(
        record_id=row["record_id"],
        pdf_path=Path(row["pdf_path"]),
        xml_path=Path(row["xml_path"]),
        tei_path=output_dir / "tei" / parser / f"{row['record_id']}.tei.xml",
        alto_path=output_dir / "alto" / f"{row['record_id']}.alto.xml",
        prediction_path=output_dir / "predictions" / parser / f"{row['record_id']}.json",
    )


def build_document_context(paths: DocumentPaths) -> DocumentContext:
    lines = extract_alto_lines(paths.alto_path)
    return DocumentContext(
        record_id=paths.record_id,
        header_text=read_tei_header(paths.tei_path),
        lines=lines,
        first_page_lines=[line for line in lines if line.get("page", 0) == 0],
        tei_fields=extract_tei_fields(paths.tei_path),
        tei_abstracts=[normalize_whitespace(text) for text in extract_tei_abstracts(paths.tei_path)],
    )


def format_header_lines(lines: Sequence[LayoutLine]) -> str:
    return "\n".join(
        f"{index + 1:02d} | y={line['y']:.1f} x={line['x']:.1f} | {line['text']}" for index, line in enumerate(lines)
    )


def validate_tei_metadata(metadata: MetadataRecord, header_text: str) -> List[str]:
    errors: List[str] = []
    if "<title" in header_text and not metadata.get("title"):
        errors.append("title missing")
    if "<author" in header_text and not metadata.get("authors"):
        errors.append("authors missing")
    if "<abstract" in header_text:
        abstract = str(metadata.get("abstract", ""))
        if not abstract or len(abstract) < 40:
            errors.append("abstract missing or too short")
    return errors


def marker_windows(
    lines: Sequence[LayoutLine],
    *,
    max_blocks: int,
    prefix_lines: int,
    suffix_lines: int,
    fallback_lines: int,
) -> List[str]:
    # Search the early PDF pages, not just page 0/1: SciELO preprint cover
    # sheets can push the article's real abstract to page 2 or 3. The heading
    # match below still keeps reference-list URLs out.
    front_page_limit = 5

    def _on_front(line: LayoutLine) -> bool:
        return _layout_line_page(line) <= front_page_limit

    indices = [
        index for index, line in enumerate(lines)
        if _on_front(line)
        and ABSTRACT_HEADING_RE.match((line.get("text", "") or "").strip()[:80])
    ]
    if not indices:
        # OCR-mash fallback: line on early pages STARTING with the
        # keyword (after optional section number), so a URL like
        # "v1.abstract" embedded mid-line cannot match.
        indices = [
            index for index, line in enumerate(lines)
            if _on_front(line)
            and re.match(
                r"^\s*(?:[\dIVXivx]+(?:\.\d+)*[\.\)]?\s+)?(abstract|resumo|resumen)\b",
                line.get("text", "") or "",
                re.IGNORECASE,
            )
        ]
    blocks: List[str] = []
    if indices:
        for index in indices[:max_blocks]:
            start = max(0, index - prefix_lines)
            tail_cap = min(len(lines), index + suffix_lines)
            end = tail_cap
            for j in range(index + 1, tail_cap):
                if NEXT_SECTION_RE.match(lines[j].get("text", "") or ""):
                    end = j
                    break
            parts: List[str] = []
            for offset, line in enumerate(lines[start:end], start=start):
                text = line.get("text", "")
                if (
                    offset == index
                    and ABSTRACT_HEADING_RE.match(text.strip()[:80])
                    and not text.rstrip().endswith((".", ":"))
                ):
                    text = text.rstrip() + ":"
                parts.append(text)
            text = " ".join(parts)
            blocks.append(normalize_whitespace(text))
    else:
        text = " ".join(line["text"] for line in lines[:fallback_lines])
        blocks.append(normalize_whitespace(text))
    return blocks


def _word_count(text: str) -> int:
    return len(TEXT_TOKEN_RE.findall(text or ""))


def is_boilerplate_candidate(text: str) -> bool:
    normalized = normalize_whitespace(text)
    if not normalized:
        return False
    if re.match(r"^\s*(abstract|resumo|resumen)\b", normalized, re.IGNORECASE) and _word_count(normalized) >= 40:
        return False
    return bool(DISCLAIMER_RE.search(normalized))


def _dedupe_candidate_texts(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        text = normalize_whitespace(value)
        key = text.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _sentence_chunks(text: str) -> List[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÃÕÇÑ])", text) if part.strip()]
    return parts if len(parts) > 1 else [normalize_whitespace(text)]


def split_abstract_language_segments(text: str, min_words: int = 35) -> List[str]:
    normalized = normalize_whitespace(text)
    if _word_count(normalized) < min_words * 2:
        return []

    candidates: List[str] = []
    marker_parts = [
        part.strip()
        for part in re.split(r"(?=\b(?:abstract|resumo|resumen)\b\s*:?)", normalized, flags=re.IGNORECASE)
        if part.strip()
    ]
    if len(marker_parts) > 1:
        candidates.extend(part for part in marker_parts if _word_count(part) >= min_words)

    if not is_mixed_language(normalized):
        return _dedupe_candidate_texts(candidates)

    chunks = _sentence_chunks(normalized)
    if len(chunks) < 2:
        return _dedupe_candidate_texts(candidates)

    word_counts = [_word_count(chunk) for chunk in chunks]
    total_words = sum(word_counts)
    left_words = 0
    best_split: Optional[Tuple[str, str, float]] = None
    for index in range(1, len(chunks)):
        left_words += word_counts[index - 1]
        right_words = total_words - left_words
        if left_words < min_words or right_words < min_words:
            continue
        left = normalize_whitespace(" ".join(chunks[:index]))
        right = normalize_whitespace(" ".join(chunks[index:]))
        left_language = detect_language(left)
        right_language = detect_language(right)
        if left_language == "unknown" or right_language == "unknown" or left_language == right_language:
            continue
        balance = min(left_words, right_words) / max(left_words, right_words)
        score = balance + (0.25 if right_language == "en" else 0.0) + (0.1 if left_language == "en" else 0.0)
        if best_split is None or score > best_split[2]:  # pylint: disable=unsubscriptable-object
            best_split = (left, right, score)
    if best_split:
        candidates.extend([best_split[0], best_split[1]])
    return _dedupe_candidate_texts(candidates)


def expand_abstract_candidate(source: str, text: str) -> List[Tuple[str, str]]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    blocks = [(source, normalized)]
    for index, segment in enumerate(split_abstract_language_segments(normalized), start=1):
        if segment.casefold() != normalized.casefold():
            blocks.append((f"{source}_segment_{index}", segment))
    return blocks


def dedupe_tagged_blocks(blocks: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    unique: List[Tuple[str, str]] = []
    for source, text in blocks:
        key = normalize_whitespace(text).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((source, text))
    return unique


def _block_tokens(text: str) -> set:
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text or "")
    # Drop pure-numeric tokens — typeset PDFs leak line numbers ("23", "24") into
    # the OCR copy of the abstract, defeating subset checks against the clean copy.
    return {t for t in re.findall(r"\w+", nfkd.lower()) if not t.isdigit()}


def dedupe_blocks(blocks: Sequence[str], near_subset_ratio: float = 0.9) -> List[str]:
    # Drop empties, exact duplicates, blocks that are substrings of another,
    # and blocks whose NFKD-normalised token set is contained (strictly or
    # near-fully, default >=90%) in another block's tokens. The fuzzy bound
    # catches OCR copies that share most words with the clean abstract but
    # have a few unique tokens (heading words, line numbers stripped earlier,
    # truncations).
    keys = [normalize_whitespace(text).lower() for text in blocks]
    token_sets = [_block_tokens(text) for text in blocks]
    seen: set[str] = set()
    unique: List[str] = []

    def _near_contained(small: set, big: set) -> bool:
        if not small:
            return False
        if len(big) <= len(small):
            return False
        return len(small & big) / len(small) >= near_subset_ratio

    for i, (text, key) in enumerate(zip(blocks, keys)):
        if not key or key in seen:
            continue
        if any(key != other and key in other for other in keys):
            continue
        ti = token_sets[i]
        if ti and any(
            i != j and _near_contained(ti, token_sets[j])
            for j in range(len(blocks))
        ):
            continue
        seen.add(key)
        unique.append(text)
    return unique


def build_abstract_candidates(context: DocumentContext) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    for text in context.tei_abstracts:
        if not is_boilerplate_candidate(text):
            blocks.extend(expand_abstract_candidate("tei_abstract", text))
    for index, block in enumerate(
        marker_windows(
            context.lines,
            max_blocks=3,
            prefix_lines=0,
            suffix_lines=80,
            fallback_lines=120,
        ),
        start=1,
    ):
        blocks.extend(expand_abstract_candidate(f"alto_block_{index}", block))
    return dedupe_tagged_blocks(blocks)


def format_candidate_blocks(
    blocks: Sequence[Tuple[str, str]],
    max_block_chars: int = 1200,
    max_total_chars: int = 8000,
) -> str:
    parts: List[str] = []
    total = 0
    for index, (source, text) in enumerate(blocks, start=1):
        chunk = f"[{index}] source={source}\n{text[:max_block_chars]}\n"
        if total + len(chunk) > max_total_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


def build_multilingual_abstract_blocks(context: DocumentContext) -> List[str]:
    blocks: List[str] = []
    for text in context.tei_abstracts:
        if not is_boilerplate_candidate(text):
            blocks.append(text)
    for block in marker_windows(
        context.lines,
        max_blocks=4,
        prefix_lines=0,
        suffix_lines=18,
        fallback_lines=160,
    ):
        if not is_boilerplate_candidate(block):
            blocks.append(block)
    return dedupe_blocks(blocks)


def build_ocr_input(lines: Sequence[LayoutLine], max_lines: int = 200) -> str:
    return "\n".join(line["text"] for line in lines[:max_lines] if line.get("text"))


def slice_near_abstract_marker(text: str, max_chars: int = 5000) -> str:
    match = ABSTRACT_MARKER_RE.search(text)
    if not match:
        return text[:max_chars]
    start = max(0, match.start() - 200)
    return text[start : start + max_chars]


def score_abstract_candidate(text: str) -> float:
    if not text or not text.strip():
        return float("-inf")
    word_count = len(re.findall(r"\w+", text))
    score = min(word_count / 200.0, 1.0)
    if is_boilerplate_candidate(text):
        score -= 5.0
    if ENGLISH_START_RE.search(text):
        score += 2.0
    if ROMANCE_START_RE.search(text):
        score -= 1.0
    # Prefer full-length abstracts (50-500 words covers most journals); under 30
    # is almost certainly truncated; over 500 is bleeding past the abstract.
    if 50 <= word_count <= 500:
        score += 1.0
    if word_count < 30:
        score -= 2.0
    if word_count > 800:
        score -= 1.0
    return score


def _support_key(text: str) -> str:
    return normalize_whitespace(text).casefold()


def _support_tokens(text: str) -> List[str]:
    return [token for token in TEXT_TOKEN_RE.findall(_support_key(text)) if not token.isdigit()]


def is_extractively_supported(text: str, sources: Sequence[str]) -> bool:
    candidate = _support_key(text)
    if not candidate:
        return False
    for source in sources:
        source_key = _support_key(source)
        if not source_key:
            continue
        if candidate in source_key:
            return True
        # Allow light OCR whitespace/hyphenation repair while rejecting summaries.
        if len(candidate) >= 80 and SequenceMatcher(None, candidate, source_key).ratio() >= 0.92:
            return True
        candidate_tokens = _support_tokens(candidate)
        if len(candidate_tokens) >= 20:
            source_tokens = set(_support_tokens(source_key))
            if source_tokens:
                overlap = sum(1 for token in candidate_tokens if token in source_tokens)
                if overlap / len(candidate_tokens) >= 0.9:
                    return True
    return False


def require_extractive_support(text: str, sources: Sequence[str]) -> str:
    normalized = normalize_whitespace(text)
    if not normalized:
        return ""
    return normalized if is_extractively_supported(normalized, sources) else ""


def prefer_unmixed_abstract_candidate(text: str, candidates: Sequence[str]) -> str:
    selected = normalize_whitespace(text)
    if not selected:
        return selected
    available = [
        normalize_whitespace(candidate)
        for candidate in candidates
        if candidate and not is_boilerplate_candidate(candidate) and not is_mixed_language(candidate)
    ]
    if not available:
        return selected
    english = [candidate for candidate in available if detect_language(candidate) == "en"]
    preferred_pool = english or available
    best = max(preferred_pool, key=score_abstract_candidate)
    selected_words = _word_count(selected)
    best_words = _word_count(best)
    if is_mixed_language(selected):
        return best
    if (
        english
        and best_words >= 40
        and selected_words > best_words * 1.35
        and is_extractively_supported(best, [selected])
    ):
        return best
    return selected


def detect_language(text: str) -> str:
    tokens = [token.lower() for token in WORD_RE.findall(text or "")]
    if not tokens:
        return "unknown"
    scores = {
        language: sum(token in stopwords for token in tokens) for language, stopwords in LANGUAGE_STOPWORDS.items()
    }
    best_language, best_score = max(scores.items(), key=lambda item: item[1])
    return best_language if best_score > 0 else "unknown"


def language_scores(text: str) -> Dict[str, int]:
    tokens = [token.lower() for token in WORD_RE.findall(text or "")]
    if not tokens:
        return {lang: 0 for lang in LANGUAGE_STOPWORDS}
    return {language: sum(token in stopwords for token in tokens) for language, stopwords in LANGUAGE_STOPWORDS.items()}


def is_mixed_language(text: str) -> bool:
    scores = language_scores(text)
    ordered = sorted(scores.values(), reverse=True)
    if not ordered or ordered[0] == 0:
        return False
    top, second = ordered[0], ordered[1] if len(ordered) > 1 else 0
    return second >= 3 and second / max(1, top) >= 0.5


def canonical_language_code(language: str) -> str:
    value = (language or "").strip().lower()
    if value in {"pt", "por"}:
        return "pt"
    if value in {"en", "eng"}:
        return "en"
    if value in {"es", "spa"}:
        return "es"
    return value or "unknown"


def split_title_candidates(title: str, preferred_language: Optional[str]) -> List[str]:
    if not title:
        return []
    text = normalize_whitespace(title)
    candidates: List[str] = []

    # Hard split on newlines first.
    newline_parts = [part.strip() for part in re.split(r"[\r\n]+", title) if part.strip()]
    if len(newline_parts) > 1:
        candidates.extend(newline_parts)

    # Split on explicit separators.
    for pattern in (r"\s+/\s+", r"\s+\|\s+", r"\s+—\s+", r"\s+–\s+"):
        parts = [part.strip() for part in re.split(pattern, text) if part.strip()]
        if len(parts) > 1:
            candidates.extend(parts)

    # Split on repeated prefix (e.g., duplicated title in another language).
    tokens = text.split()
    if len(tokens) >= 6:
        prefix = " ".join(tokens[:3]).lower()
        idx = text.lower().find(prefix, len(prefix) + 1)
        if idx > 0:
            left = text[:idx].strip()
            right = text[idx:].strip()
            if left and right:
                candidates.extend([left, right])

    # Attempt a language boundary split when both languages are present.
    if len(tokens) >= 10:
        for i in range(5, len(tokens) - 5):
            left = " ".join(tokens[:i])
            right = " ".join(tokens[i:])
            left_lang = detect_language(left)
            right_lang = detect_language(right)
            if left_lang == "unknown" or right_lang == "unknown":
                continue
            if left_lang == right_lang:
                continue
            if preferred_language and preferred_language != "unknown":
                if left_lang == preferred_language:
                    candidates.extend([left, right])
                    break
                if right_lang == preferred_language:
                    candidates.extend([right, left])
                    break
            else:
                candidates.extend([left, right])
                break

    # Always include the original title as a fallback.
    candidates.append(text)

    # Deduplicate while preserving order.
    seen = set()
    unique: List[str] = []
    for candidate in candidates:
        normalized = normalize_whitespace(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def choose_title_candidate(title: str, preferred_language: Optional[str]) -> str:
    text = normalize_whitespace(title)
    # If the title does not look bilingual, keep it intact to avoid truncation.
    if not is_mixed_language(text):
        return text
    candidates = split_title_candidates(title, preferred_language)
    if not candidates:
        return text
    if preferred_language and preferred_language != "unknown":
        lang_candidates = [c for c in candidates if detect_language(c) == preferred_language]
        non_mixed = [c for c in lang_candidates if not is_mixed_language(c)]
        if non_mixed:
            return max(non_mixed, key=len)
        if lang_candidates:
            return max(lang_candidates, key=len)
    non_mixed = [c for c in candidates if not is_mixed_language(c)]
    if non_mixed:
        return max(non_mixed, key=len)
    # Fallback: choose the longest candidate (usually the most complete title).
    return max(candidates, key=len)


def detect_languages_in_lines(lines: Sequence[LayoutLine]) -> List[str]:
    languages = set()
    for line in lines[:200]:
        text = line.get("text", "")
        if ENGLISH_MARKER_RE.search(text):
            languages.add("en")
        if PORTUGUESE_MARKER_RE.search(text):
            languages.add("pt")
        if SPANISH_MARKER_RE.search(text):
            languages.add("es")
    return sorted(languages)


def choose_abstract_candidate(candidates: Sequence[str], preferred_language: Optional[str]) -> str:
    available = [candidate for candidate in candidates if candidate and candidate.strip()]
    if not available:
        return ""
    if any(not is_boilerplate_candidate(candidate) for candidate in available):
        available = [candidate for candidate in available if not is_boilerplate_candidate(candidate)]
    if any(_word_count(candidate) >= 30 for candidate in available):
        available = [candidate for candidate in available if _word_count(candidate) >= 15]
    non_mixed = [candidate for candidate in available if not is_mixed_language(candidate)]
    if any(_word_count(candidate) >= 30 for candidate in non_mixed):
        available = non_mixed
    if preferred_language and preferred_language != "unknown":
        matching = [candidate for candidate in available if detect_language(candidate) == preferred_language]
        if matching:
            return max(matching, key=score_abstract_candidate)
    return max(available, key=score_abstract_candidate)


def _abstract_candidates_agree(left: str, right: str) -> bool:
    left_key = _support_key(left)
    right_key = _support_key(right)
    if not left_key or not right_key:
        return False
    shorter, longer = (left_key, right_key) if len(left_key) <= len(right_key) else (right_key, left_key)
    if len(shorter) >= 80 and shorter in longer:
        return True
    if len(shorter) >= 80 and SequenceMatcher(None, shorter, longer).ratio() >= 0.86:
        return True
    left_tokens = _support_tokens(left_key)
    right_tokens = _support_tokens(right_key)
    if len(left_tokens) < 20 or len(right_tokens) < 20:
        return False
    small, large = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    large_set = set(large)
    return sum(1 for token in small if token in large_set) / len(small) >= 0.78


def choose_abstract_candidate_from_sources(
    candidates: Sequence[Tuple[str, str]],
    preferred_language: Optional[str],
) -> str:
    available: List[Tuple[str, str]] = []
    seen = set()
    for source, candidate in candidates:
        text = normalize_whitespace(candidate)
        key = _support_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        available.append((source, text))
    if not available:
        return ""
    if any(not is_boilerplate_candidate(candidate) for _, candidate in available):
        available = [(source, candidate) for source, candidate in available if not is_boilerplate_candidate(candidate)]
    if any(_word_count(candidate) >= 30 for _, candidate in available):
        available = [(source, candidate) for source, candidate in available if _word_count(candidate) >= 15]
    non_mixed = [(source, candidate) for source, candidate in available if not is_mixed_language(candidate)]
    if any(_word_count(candidate) >= 30 for _, candidate in non_mixed):
        available = non_mixed

    language = canonical_language_code(preferred_language or "")
    has_language_match = (
        language != "unknown"
        and any(detect_language(candidate) == language for _, candidate in available)
    )

    llm_sources = {"header_metadata", "abstract_from_candidates", "ocr_abstract"}

    def _source_family(source: str) -> str:
        if source.startswith("tei"):
            return "tei"
        return source

    def _score(item: Tuple[str, str]) -> float:
        source, candidate = item
        score = score_abstract_candidate(candidate)
        if source in llm_sources:
            score += 1.0
        elif source.startswith("alto_block"):
            score += 0.5
        elif source == "tei_fields":
            score -= 0.25
        if has_language_match:
            score += 1.0 if detect_language(candidate) == language else -0.5
        for other_source, other_candidate in available:
            if other_source == source:
                continue
            if _source_family(other_source) == _source_family(source):
                continue
            if _abstract_candidates_agree(candidate, other_candidate):
                score += 1.25
        return score

    return max(available, key=_score)[1]


_ORCID_RE = re.compile(r"^\s*(?:https?://orcid\.org/|orcid[:\s]*)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])\s*$", re.IGNORECASE)
_DOI_VALUE_RE = re.compile(r"10\.\d{4,9}/[^\s<>\"'\\)]+", re.IGNORECASE)


def _looks_like_orcid(value: str) -> bool:
    return bool(_ORCID_RE.match(value or ""))


def add_scielo_identifiers(record_id: str, identifiers: Sequence[str]) -> List[str]:
    # The LLM tends to dump every ID-shaped string from the front page (author
    # ORCIDs, etc.) into the identifiers list. Drop ORCIDs explicitly — they
    # identify a person, not the article.
    values = [value for value in identifiers if value and not _looks_like_orcid(value)]
    match = SCIELO_RECORD_RE.search(record_id)
    if match:
        identifier = match.group(1)
        values.extend(
            [
                SCIELO_VIEW_URL.format(id=identifier),
                SCIELO_DOI.format(id=identifier),
            ]
        )
    seen = set()
    deduped: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def scielo_identifiers_from_record_id(record_id: str) -> List[str]:
    match = SCIELO_RECORD_RE.search(record_id)
    if not match:
        return []
    identifier = match.group(1)
    return [
        SCIELO_VIEW_URL.format(id=identifier),
        SCIELO_DOI.format(id=identifier),
    ]


def add_scielo_landing_url(record_id: str, identifiers: Sequence[str]) -> List[str]:
    values = _dedupe_strings([str(identifier) for identifier in identifiers])
    match = SCIELO_RECORD_RE.search(record_id)
    if not match:
        return values
    url = SCIELO_VIEW_URL.format(id=match.group(1))
    if url not in values:
        values.insert(0, url)
    return values


def normalize_identifier_values(values: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        doi = _DOI_VALUE_RE.search(text)
        if doi:
            text = re.sub(r"\s+", "", doi.group(0)).rstrip(".,;:]}")
        normalized.append(text)
    return _dedupe_strings(normalized)


def has_scielo_preprint_doi_evidence(context: DocumentContext) -> bool:
    record_identifiers = scielo_identifiers_from_record_id(context.record_id)
    if len(record_identifiers) < 2:
        return False
    doi = record_identifiers[1].casefold()
    evidence = " ".join(
        line.get("text", "") for line in front_matter_layout_lines(context, max_lines=160)
        if line.get("text")
    ).casefold()
    return doi in evidence


def add_evidenced_scielo_preprint_doi(context: DocumentContext, identifiers: Sequence[str]) -> List[str]:
    values = add_scielo_landing_url(context.record_id, identifiers)
    record_identifiers = scielo_identifiers_from_record_id(context.record_id)
    if len(record_identifiers) < 2 or not has_scielo_preprint_doi_evidence(context):
        return values
    preprint_doi = record_identifiers[1]
    has_external_doi = any(
        identifier.lower().startswith("10.") and "scielopreprints" not in identifier.lower()
        for identifier in values
    )
    if has_external_doi and preprint_doi not in values:
        values.append(preprint_doi)
    return values


def coalesce_metadata(*records: MetadataRecord) -> MetadataRecord:
    merged = normalize_metadata({})
    for record in records:
        normalized = normalize_metadata(record)
        for field in SCALAR_FIELDS:
            if not merged[field] and normalized[field]:
                merged[field] = normalized[field]
        for field in LIST_FIELDS:
            if not merged[field] and normalized[field]:
                merged[field] = normalized[field]
    return merged


def merge_list_values(values: Sequence[str], additions: Sequence[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for value in [*values, *additions]:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def keyword_target_languages(keywords: Sequence[str], lines: Sequence[LayoutLine]) -> List[str]:
    if not keywords:
        return []
    base_language = detect_language(" ".join(keywords))
    available_languages = detect_languages_in_lines(lines)
    targets: List[str] = []
    if base_language in {"pt", "es"}:
        targets.append("en")
    elif base_language == "en":
        if "pt" in available_languages:
            targets.append("pt")
        if "es" in available_languages:
            targets.append("es")
    else:
        targets.extend(language for language in available_languages if language != "unknown")
    return sorted(set(targets))


def translate_keywords(chat: Callable[..., str], keywords: Sequence[str], target_languages: Sequence[str]) -> List[str]:
    if not keywords or not target_languages:
        return [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    messages = [
        {"role": "system", "content": KEYWORD_TRANSLATION_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"keywords": list(keywords), "target_languages": list(target_languages)},
                ensure_ascii=True,
            ),
        },
    ]
    payload = safe_extract_json(chat(messages, temperature=0.0, max_tokens=400, step_name="KEYWORD_TRANSLATE"))
    translations = payload.get("translations") or {}
    merged = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    for language in target_languages:
        merged = merge_list_values(merged, translations.get(language, []) or [])
    return merged


def _normalise_selection_key(value: str) -> str:
    return normalize_whitespace(value).casefold()


def _dedupe_strings(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        text = str(value).strip()
        key = _normalise_selection_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def normalize_keyword_values(values: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    marker = (
        r"(?:keywords?|key-words?|palavras[-\s]+chave|palabras\s+clave|"
        r"descritores?|descriptors?)"
    )
    leading_marker_re = re.compile(rf"^\s*{marker}\s*[:.]?\s+", re.IGNORECASE)
    embedded_marker_re = re.compile(rf"\s+{marker}\s*[:.]?\s+", re.IGNORECASE)
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        text = leading_marker_re.sub("", text).strip(" ;.:-")
        marker_parts = [part.strip(" ;.:-") for part in embedded_marker_re.split(text) if part.strip(" ;.:-")]
        for part in marker_parts or [text]:
            suspect = SUSPECT_KEYWORD_RE.search(part)
            if suspect:
                if suspect.start() < 50:
                    continue
                part = part[: suspect.start()].strip(" ;.:-")
            if not part:
                continue
            separator_count = part.count(",") + part.count(";")
            if separator_count >= 3 or part.count(";") >= 2:
                normalized.extend(item.strip(" ;.:-") for item in re.split(r"\s*[;,]\s*", part) if item.strip(" ;.:-"))
            else:
                normalized.append(part)
    normalized = [
        value for value in normalized
        if _normalise_selection_key(value) not in {"none", "n/a", "not applicable"}
    ]
    return _dedupe_strings(normalized)


def _candidate_sets_payload(candidate_sets: Sequence[Tuple[str, Sequence[str]]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for source, values in candidate_sets:
        items = _dedupe_strings([str(value) for value in values])
        if items:
            payload.append({"source": source, "values": items})
    return payload


def _filter_llm_selection_to_candidates(
    selected_values: Sequence[str],
    candidate_sets: Sequence[Tuple[str, Sequence[str]]],
) -> List[str]:
    by_key: Dict[str, str] = {}
    for _, values in candidate_sets:
        for value in values:
            text = str(value).strip()
            key = _normalise_selection_key(text)
            if key and key not in by_key:
                by_key[key] = text
    out: List[str] = []
    seen = set()
    for value in selected_values:
        key = _normalise_selection_key(str(value))
        candidate = by_key.get(key)
        if not candidate or key in seen:
            continue
        out.append(candidate)
        seen.add(key)
    return out


def _selection_overlaps_source(
    selected_values: Sequence[str],
    candidate_sets: Sequence[Tuple[str, Sequence[str]]],
    source_prefix: str,
) -> bool:
    selected_keys = {_normalise_selection_key(str(value)) for value in selected_values}
    selected_keys.discard("")
    if not selected_keys:
        return False
    for source, values in candidate_sets:
        if not source.startswith(source_prefix):
            continue
        source_keys = {_normalise_selection_key(str(value)) for value in values}
        if selected_keys & source_keys:
            return True
    return False


def _keyword_marker_evidence_text(context: DocumentContext, max_lines: int = 220) -> str:
    lines = front_matter_evidence_lines(context, max_lines=max_lines, drop_identifiers=True)
    indices = [index for index, line in enumerate(lines) if KEYWORD_MARKER_RE.search(line.get("text", "") or "")]
    if not indices:
        return ""
    snippets: List[str] = []
    for index in indices[:4]:
        start = max(0, index - 1)
        end = min(len(lines), index + 5)
        snippets.extend(line.get("text", "") for line in lines[start:end] if line.get("text"))
    return normalize_whitespace(" ".join(snippets))


def _keyword_selection_has_explicit_evidence(context: DocumentContext, selected_values: Sequence[str]) -> bool:
    evidence = _support_key(_keyword_marker_evidence_text(context))
    if not evidence:
        return False
    matched = 0
    for value in selected_values:
        key = _normalise_selection_key(str(value))
        if key and key in evidence:
            matched += 1
    return matched >= min(2, len([value for value in selected_values if str(value).strip()]))


def _keyword_selection_needs_explicit_evidence(selected_values: Sequence[str]) -> bool:
    values = [str(value).strip() for value in selected_values if str(value).strip()]
    if len(values) > 12:
        return True
    for value in values:
        if SUSPECT_KEYWORD_RE.search(value):
            return True
    if len(values) == 1:
        value = values[0]
        # Period-delimited TEI terms are usually a collapsed keyword list, not
        # one author keyword. Keep abbreviations like COVID-19 intact.
        if len([part for part in re.split(r"\.\s+", value) if part.strip()]) >= 3:
            return True
    return False


def _preferred_front_matter_keywords(
    candidate_sets: Sequence[Tuple[str, Sequence[str]]],
    preferred_language: Optional[str],
) -> List[str]:
    front_sets: List[Tuple[str, List[str]]] = [
        (source, _dedupe_strings([str(value) for value in values]))
        for source, values in candidate_sets
        if source.startswith("front_matter_llm")
    ]
    front_sets = [(source, values) for source, values in front_sets if values]
    if not front_sets:
        return []
    language = canonical_language_code(preferred_language or "")
    for source, values in front_sets:
        parts = source.split(":")
        if len(parts) >= 2 and canonical_language_code(parts[1]) == "en":
            return values
    if language != "unknown":
        for source, values in front_sets:
            parts = source.split(":")
            if len(parts) >= 2 and canonical_language_code(parts[1]) == language:
                return values
    return front_sets[0][1]


def validate_keyword_selection(
    context: DocumentContext,
    selected_values: Sequence[str],
    candidate_sets: Sequence[Tuple[str, Sequence[str]]],
) -> List[str]:
    values = _dedupe_strings([str(value) for value in selected_values])
    if not values:
        return []
    # Front-matter extraction has already been grounded in explicit keyword
    # evidence. For TEI/header-only selections, require independent marker
    # support so navigation topics or funding statements cannot survive just
    # because they are the only candidate list.
    if _selection_overlaps_source(values, candidate_sets, "front_matter_llm"):
        return values
    if _keyword_selection_has_explicit_evidence(context, values):
        return values
    if not _keyword_selection_needs_explicit_evidence(values):
        return values
    return []


def select_keywords_from_candidates(
    context: DocumentContext,
    chat: Callable[..., str],
    candidate_sets: Sequence[Tuple[str, Sequence[str]]],
    *,
    title: str,
    abstract: str,
    preferred_language: Optional[str],
) -> List[str]:
    payload_sets = _candidate_sets_payload(candidate_sets)
    if not payload_sets:
        return []
    if len(payload_sets) == 1 and str(payload_sets[0].get("source", "")).startswith("front_matter_llm"):
        return [str(value) for value in payload_sets[0]["values"]]
    front_matter_values = _preferred_front_matter_keywords(candidate_sets, preferred_language)

    messages = [
        {"role": "system", "content": KEYWORD_SELECTION_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "record_id": context.record_id,
                    "language": canonical_language_code(preferred_language or ""),
                    "title": title,
                    "selected_abstract": abstract[:2000],
                    "candidate_keyword_lists": payload_sets,
                    "front_matter_lines": [
                        line.get("text", "") for line in front_matter_evidence_lines(
                            context,
                            max_lines=120,
                            drop_identifiers=True,
                        )
                        if line.get("text")
                    ],
                },
                ensure_ascii=True,
            ),
        },
    ]
    try:
        payload = safe_extract_json(chat(messages, temperature=0.0, max_tokens=500, step_name="KEYWORD_SELECT"))
    except Exception:
        payload = {}
    selected = payload.get("keywords")
    if isinstance(selected, list):
        filtered = _filter_llm_selection_to_candidates(selected, candidate_sets)
        validated = validate_keyword_selection(context, filtered, candidate_sets)
        front_matter_is_safe = (
            bool(front_matter_values)
            and (
                not _keyword_selection_needs_explicit_evidence(front_matter_values)
                or _keyword_selection_has_explicit_evidence(context, front_matter_values)
            )
        )
        if front_matter_is_safe and len(validated) < len(front_matter_values):
            return front_matter_values
        return validated
    return validate_keyword_selection(context, [str(value) for value in payload_sets[0]["values"]], candidate_sets)


def extract_keyword_candidate_sets_from_front_matter(
    context: DocumentContext,
    chat: Callable[..., str],
    *,
    title: str,
    abstract: str,
    max_lines: int = 220,
) -> List[Tuple[str, List[str]]]:
    lines = [
        line.get("text", "") for line in front_matter_evidence_lines(
            context,
            max_lines=max_lines,
            drop_identifiers=True,
        )
        if line.get("text")
    ]
    if not lines or not _keyword_marker_evidence_text(context, max_lines=max_lines):
        return []
    messages = [
        {"role": "system", "content": KEYWORD_EXTRACTION_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "record_id": context.record_id,
                    "title": title,
                    "selected_abstract": abstract[:2000],
                    "front_matter_lines": lines,
                },
                ensure_ascii=True,
            ),
        },
    ]
    try:
        payload = safe_extract_json(chat(messages, temperature=0.0, max_tokens=700, step_name="KEYWORD_EXTRACT"))
    except Exception:
        return []
    lists = payload.get("keyword_lists") or []
    out: List[Tuple[str, List[str]]] = []
    if isinstance(lists, list):
        for index, item in enumerate(lists, start=1):
            if not isinstance(item, dict):
                continue
            keywords = item.get("keywords") or []
            if not isinstance(keywords, list):
                continue
            values = normalize_keyword_values([str(keyword) for keyword in keywords])
            if not values:
                continue
            language = canonical_language_code(str(item.get("language") or "unknown"))
            out.append((f"front_matter_llm:{language}:{index}", values))
    return out


def infer_keywords_from_metadata(
    context: DocumentContext,
    chat: Callable[..., str],
    *,
    title: str,
    abstract: str,
    max_lines: int = 90,
) -> List[str]:
    if not normalize_whitespace(title) and _word_count(abstract) < 30:
        return []
    lines = [
        line.get("text", "") for line in front_matter_evidence_lines(
            context,
            max_lines=max_lines,
            drop_identifiers=True,
        )
        if line.get("text")
    ]
    messages = [
        {"role": "system", "content": KEYWORD_INFERENCE_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "record_id": context.record_id,
                    "title": title,
                    "selected_abstract": abstract[:3000],
                    "front_matter_lines": lines,
                },
                ensure_ascii=True,
            ),
        },
    ]
    try:
        payload = safe_extract_json(chat(messages, temperature=0.0, max_tokens=500, step_name="KEYWORD_INFER"))
    except Exception:
        return []
    keywords = payload.get("keywords") or []
    if not isinstance(keywords, list):
        return []
    values = normalize_keyword_values([str(keyword) for keyword in keywords])
    values = [
        value for value in values
        if 1 <= _word_count(value) <= 8 and not SUSPECT_KEYWORD_RE.search(value)
    ]
    return _dedupe_strings(values[:8])


def select_article_identifiers_from_candidates(
    context: DocumentContext,
    chat: Callable[..., str],
    candidate_sets: Sequence[Tuple[str, Sequence[str]]],
    *,
    title: str,
) -> List[str]:
    candidate_sets = [(source, normalize_identifier_values(values)) for source, values in candidate_sets]
    payload_sets = _candidate_sets_payload(candidate_sets)
    if not payload_sets:
        return []
    messages = [
        {"role": "system", "content": IDENTIFIER_SELECTION_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "record_id": context.record_id,
                    "title": title,
                    "candidate_identifier_lists": payload_sets,
                    "front_matter_lines": [
                        line.get("text", "") for line in front_matter_layout_lines(context, max_lines=120)
                        if line.get("text")
                    ],
                },
                ensure_ascii=True,
            ),
        },
    ]
    try:
        payload = safe_extract_json(chat(messages, temperature=0.0, max_tokens=400, step_name="IDENTIFIER_SELECT"))
    except Exception:
        payload = {}
    selected = payload.get("identifiers") or []
    if isinstance(selected, list):
        filtered = _filter_llm_selection_to_candidates(selected, candidate_sets)
        filtered = [value for value in filtered if not _looks_like_orcid(value)]
        if filtered:
            return add_evidenced_scielo_preprint_doi(context, filtered)

    record_identifiers = scielo_identifiers_from_record_id(context.record_id)
    if record_identifiers:
        return record_identifiers

    fallback: List[str] = []
    for _, values in candidate_sets:
        for value in values:
            text = str(value).strip()
            if text and not _looks_like_orcid(text):
                fallback.append(text)
    return _dedupe_strings(fallback[:2])


def _estimate_column_count(
    lines: Sequence[LayoutLine],
    *,
    min_lines_per_col: int = 3,
    gap_threshold: float = 80.0,
) -> int:
    if len(lines) < 2 * min_lines_per_col:
        return 1
    xs = sorted(float(line.get("x", 0.0) or 0.0) for line in lines)
    gaps = [
        (xs[index + 1] - xs[index], (xs[index] + xs[index + 1]) / 2.0)
        for index in range(len(xs) - 1)
    ]
    if not gaps:
        return 1
    gap, split_x = max(gaps, key=lambda item: item[0])
    if gap < gap_threshold:
        return 1
    left = sum(1 for line in lines if float(line.get("x", 0.0) or 0.0) <= split_x)
    right = len(lines) - left
    if left < min_lines_per_col or right < min_lines_per_col:
        return 1
    return 2


_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s+(\w)")


def _dehyphenate(text: str) -> str:
    """Collapse end-of-line hyphenation when joining ALTO lines: 'experi- ment' -> 'experiment'.
    Conservative: only joins when both sides are word characters.
    """
    prev = None
    while prev != text:
        prev = text
        text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    return text


def resolve_field_text(
    parsed_text: str,
    indices: Sequence[Any],
    lines: Sequence[LayoutLine],
) -> str:
    """Reconstruct field text from LLM-supplied 1-indexed line indices into ALTO lines.

    Returns the LLM's parsed_text fallback when:
      - indices is empty/missing
      - any index is out of bounds, non-integer, or duplicated
      - the indexed lines have no usable text content

    Otherwise joins the referenced lines in the order given (preserving the
    LLM's chosen sequence), de-hyphenates trailing-hyphen line breaks, and
    collapses whitespace.
    """
    if not indices:
        return parsed_text
    valid: List[int] = []
    seen: set = set()
    for raw in indices:
        if isinstance(raw, bool) or not isinstance(raw, int):
            return parsed_text
        if not 1 <= raw <= len(lines):
            return parsed_text
        if raw in seen:
            return parsed_text
        seen.add(raw)
        valid.append(raw)
    pieces = [(lines[i - 1].get("text") or "").strip() for i in valid]
    pieces = [p for p in pieces if p]
    if not pieces:
        return parsed_text
    joined = " ".join(pieces)
    joined = _dehyphenate(joined)
    return re.sub(r"\s+", " ", joined).strip()


def resolve_field_list(
    parsed_items: Sequence[str],
    indices_groups: Sequence[Any],
    lines: Sequence[LayoutLine],
) -> List[str]:
    """Reconstruct each list item from its group of 1-indexed line indices.

    `indices_groups` is a list of lists; the i-th inner list holds the lines
    that comprise the i-th item. Each inner group is reconstructed via the
    same validation + dehyphenation as `resolve_field_text`. On any inner
    failure, falls back to the LLM's `parsed_items[i]` (or "" if missing).
    Returns the full list, preserving the LLM's order.
    """
    out: List[str] = []
    parsed_len = len(parsed_items)
    if not indices_groups:
        return [str(p) for p in parsed_items if str(p).strip()]
    for i, group in enumerate(indices_groups):
        fallback = str(parsed_items[i]) if i < parsed_len else ""
        if not isinstance(group, list):
            if fallback.strip():
                out.append(fallback)
            continue
        text = resolve_field_text(fallback, group, lines)
        if text.strip():
            out.append(text)
    return out


def predict_header_metadata(context: DocumentContext, chat: Callable[..., str]) -> MetadataRecord:
    lines = front_matter_evidence_lines(context, max_lines=80, max_page=3)
    first_page_lines = [ln for ln in lines if int(ln.get("page", 0) or 0) == 0]
    if _estimate_column_count(first_page_lines) == 1:
        lines = sorted(
            lines,
            key=lambda item: (
                int(item.get("page", 0) or 0),
                float(item.get("y", 0.0) or 0.0),
                float(item.get("x", 0.0) or 0.0),
            ),
        )
    messages = [
        {"role": "system", "content": HEADER_METADATA_PROMPT},
        {"role": "user", "content": format_header_lines(lines)},
    ]
    raw = chat(messages, temperature=0.0, max_tokens=1100, step_name="HEADER_METADATA")
    parsed = safe_extract_json(raw)
    metadata = normalize_metadata(parsed)
    metadata["title"] = resolve_field_text(
        metadata.get("title", "") or "", parsed.get("title_lines") or [], lines
    )
    metadata["abstract"] = resolve_field_text(
        metadata.get("abstract", "") or "", parsed.get("abstract_lines") or [], lines
    )
    parsed_authors = parsed.get("authors") or []
    if isinstance(parsed_authors, list):
        author_groups = parsed.get("author_groups") or []
        if isinstance(author_groups, list) and author_groups:
            metadata["authors"] = resolve_field_list(
                [str(a) for a in parsed_authors], author_groups, lines
            )
    parsed_keywords = parsed.get("keywords") or []
    if isinstance(parsed_keywords, list):
        keyword_groups = parsed.get("keyword_groups") or []
        if isinstance(keyword_groups, list) and keyword_groups:
            metadata["keywords"] = resolve_field_list(
                [str(k) for k in parsed_keywords], keyword_groups, lines
            )
    return metadata


def predict_tei_metadata(context: DocumentContext, chat: Callable[..., str]) -> MetadataRecord:
    messages = [
        {"role": "system", "content": TEI_METADATA_PROMPT},
        {"role": "user", "content": context.header_text},
    ]
    raw = chat(messages, temperature=0.0, max_tokens=900, step_name="TEI_METADATA")
    return normalize_metadata(safe_extract_json(raw))


def predict_validated_tei_metadata(context: DocumentContext, chat: Callable[..., str]) -> MetadataRecord:
    return _predict_tei_metadata_with_validation(context, chat)[1]


def _predict_tei_metadata_with_validation(
    context: DocumentContext, chat: Callable[..., str]
) -> Tuple[MetadataRecord, MetadataRecord]:
    # Share the TEI_METADATA first-attempt call between the raw and validated
    # predictions; only fire TEI_VALIDATED retries if validation actually fails.
    first = predict_tei_metadata(context, chat)
    errors = validate_tei_metadata(first, context.header_text)
    if not errors:
        return first, first
    current = first
    attempts = 1
    while errors and attempts < 3:
        messages = [
            {"role": "system", "content": TEI_METADATA_PROMPT},
            {"role": "user", "content": context.header_text},
            {
                "role": "user",
                "content": "Validation errors: " + ", ".join(errors) + ". Correct the JSON using only the TEI snippet.",
            },
        ]
        current = normalize_metadata(
            safe_extract_json(chat(messages, temperature=0.0, max_tokens=900, step_name="TEI_VALIDATED"))
        )
        errors = validate_tei_metadata(current, context.header_text)
        attempts += 1
    return first, current


def select_abstract_from_candidates(context: DocumentContext, chat: Callable[..., str]) -> str:
    blocks = build_abstract_candidates(context)
    if not blocks:
        return ""
    messages = [
        {"role": "system", "content": ABSTRACT_SELECTION_PROMPT},
        {"role": "user", "content": format_candidate_blocks(blocks)},
    ]
    payload = safe_extract_json(chat(messages, temperature=0.0, max_tokens=600, step_name="ABSTRACT_SELECT"))
    sources = [text for _, text in blocks]
    selected = require_extractive_support(str(payload.get("abstract", "")), sources)
    return prefer_unmixed_abstract_candidate(selected, sources)


def clean_ocr_text(context: DocumentContext, chat: Callable[..., str]) -> str:
    messages = [
        {"role": "system", "content": OCR_CLEANUP_PROMPT},
        {"role": "user", "content": build_ocr_input(context.lines)[:8000]},
    ]
    return normalize_whitespace(chat(messages, temperature=0.0, max_tokens=800, step_name="OCR_CLEANUP"))


def extract_abstract_from_ocr(clean_text: str, chat: Callable[..., str]) -> str:
    messages = [
        {"role": "system", "content": ABSTRACT_EXTRACTION_PROMPT},
        {"role": "user", "content": clean_text[:8000]},
    ]
    payload = safe_extract_json(chat(messages, temperature=0.0, max_tokens=600, step_name="ABSTRACT_FROM_OCR"))
    return require_extractive_support(str(payload.get("abstract", "")), [clean_text])


_CONTENT_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s<>\"'\\)]+", re.IGNORECASE)

_FIGURE_CAPTION_START_RE = re.compile(
    r"^\s*(figure|fig|figura|esquema)\s*\.?\s*(?:s\s*)?(?:\d+[A-Za-z]?|[IVX]+)\b",
    re.IGNORECASE,
)
_TABLE_CAPTION_START_RE = re.compile(
    r"^\s*(?:table|tabla|tabela|tab)\s*\.?\s*(?:s\s*)?(?:\d+[A-Za-z]?|[IVX]+)\s*(?:[\.:;\-–—]|$)",
    re.IGNORECASE,
)
_BODY_SECTION_NUMBER_PREFIX_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*|[IVXivx]+)[\.\)]?\s+(.+)$"
)
_BODY_SECTION_EMBEDDED_NUMBER_RE = re.compile(
    r"\s+(?=\d+(?:\.\d+)+[\.\)]?\s+[A-ZÁÉÍÓÚÃÕÇÑ])"
)


def _is_figure_caption(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < 10 or len(s) > 5000:
        return False
    return bool(_FIGURE_CAPTION_START_RE.match(s))


def _is_table_caption(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < 10 or len(s) > 5000:
        return False
    return bool(_TABLE_CAPTION_START_RE.match(s))


def _table_caption_text_is_complete(parts: Sequence[str]) -> bool:
    caption = normalize_whitespace(" ".join(parts))
    return _word_count(caption) >= 5 and bool(re.search(r"[\.\!\?\)]\s*$", caption))


def _looks_like_figure_content_after_caption(
    text: str,
    line: LayoutLine,
    start_line: LayoutLine,
    *,
    parts: Sequence[str],
    same_baseline_neighbors: int = 0,
) -> bool:
    if not _table_caption_text_is_complete(parts):
        return False
    s = normalize_whitespace(text)
    if not s:
        return True
    word_count = _word_count(s)
    if same_baseline_neighbors >= 2:
        return True
    if re.match(r"^\s*[A-Z](?:[,/][A-Z])?[\).:]\s+\S", s) and word_count <= 12:
        return True
    if (
        word_count <= 10
        and _line_font_size(line) >= _line_font_size(start_line) + 1.0
        and _looks_like_section_heading(s)
    ):
        return True
    return False


def _figure_caption_candidate_entries(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 300,
    max_lines_per_caption: int = 18,
) -> List[Tuple[str, LayoutLine]]:
    candidates: List[Tuple[str, LayoutLine]] = []
    seen: set[str] = set()
    consumed_indices: set[int] = set()
    same_baseline_neighbor_counts = _same_baseline_text_neighbor_count_map(lines)
    for index, line in enumerate(lines):
        if index in consumed_indices:
            continue
        text = _layout_text(line)
        if not _FIGURE_CAPTION_START_RE.match(text):
            continue
        parts = [text]
        consumed = [index]
        current_index = index
        while len(parts) < max_lines_per_caption:
            next_index = _next_same_column_index(lines, current_index, x_tolerance=42.0)
            if next_index is None or next_index in consumed_indices:
                break
            next_line = lines[next_index]
            if _layout_line_page(next_line) != _layout_line_page(line):
                break
            next_text = _layout_text(next_line)
            if not next_text:
                break
            if _FIGURE_CAPTION_START_RE.match(next_text) or _TABLE_CAPTION_START_RE.match(next_text):
                break
            if _looks_like_figure_content_after_caption(
                next_text,
                next_line,
                line,
                parts=parts,
                same_baseline_neighbors=same_baseline_neighbor_counts.get(next_index, 0),
            ):
                break
            y_gap = _line_y(next_line) - _line_y(lines[current_index])
            if y_gap <= 0 or y_gap > max(24.0, _line_height(lines[current_index]) * 2.6):
                break
            parts.append(next_text)
            consumed.append(next_index)
            current_index = next_index
        caption = normalize_whitespace(" ".join(parts))
        if _is_figure_caption(caption):
            key = caption.casefold()
            if key not in seen:
                seen.add(key)
                candidates.append((caption, line))
                consumed_indices.update(consumed)
                if len(candidates) >= max_candidates:
                    break
    return candidates


def figure_caption_candidate_texts(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 300,
) -> List[str]:
    return [text for text, _ in _figure_caption_candidate_entries(lines, max_candidates=max_candidates)]


def build_figure_caption_candidate_evidence(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 300,
    max_chars: int = 40000,
) -> str:
    parts: List[str] = []
    entries = _figure_caption_candidate_entries(lines, max_candidates=max_candidates)
    for text, line in entries:
        entry = (
            f"[{len(parts) + 1}] page={_layout_line_page(line) + 1} "
            f"y={float(line.get('y', 0.0) or 0.0):.1f} x={float(line.get('x', 0.0) or 0.0):.1f} | {text}"
        )
        if sum(len(part) + 1 for part in parts) + len(entry) > max_chars:
            break
        parts.append(entry)
    return "\n".join(parts)


def _looks_like_table_content_after_caption(
    text: str,
    line: LayoutLine,
    start_line: LayoutLine,
    *,
    parts: Sequence[str],
    same_baseline_neighbors: int = 0,
) -> bool:
    if not _table_caption_text_is_complete(parts):
        return False
    s = normalize_whitespace(text)
    if not s:
        return True
    word_count = _word_count(s)
    if same_baseline_neighbors >= 2:
        return True
    if (
        word_count <= 8
        and len(s) <= 90
        and not re.search(r"[\.\!\?]\s*$", s)
        and (
            bool(line.get("bold"))
            or _line_font_size(line) < _line_font_size(start_line) - 0.25
            or same_baseline_neighbors >= 1
        )
    ):
        return True
    if (
        word_count <= 10
        and _line_font_size(line) >= _line_font_size(start_line) + 1.0
        and _looks_like_section_heading(s)
    ):
        return True
    return False


def _table_caption_candidate_entries(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 250,
    max_lines_per_caption: int = 18,
) -> List[Tuple[str, LayoutLine]]:
    candidates: List[Tuple[str, LayoutLine]] = []
    seen: set[str] = set()
    consumed_indices: set[int] = set()
    same_baseline_neighbor_counts = _same_baseline_text_neighbor_count_map(lines)
    for index, line in enumerate(lines):
        if index in consumed_indices:
            continue
        text = _layout_text(line)
        if not _TABLE_CAPTION_START_RE.match(text):
            continue
        parts = [text]
        consumed = [index]
        current_index = index
        while len(parts) < max_lines_per_caption:
            next_index = _next_same_column_index(lines, current_index, x_tolerance=42.0)
            if next_index is None or next_index in consumed_indices:
                break
            next_line = lines[next_index]
            if _layout_line_page(next_line) != _layout_line_page(line):
                break
            next_text = _layout_text(next_line)
            if not next_text:
                break
            if _TABLE_CAPTION_START_RE.match(next_text) or _FIGURE_CAPTION_START_RE.match(next_text):
                break
            if _looks_like_table_content_after_caption(
                next_text,
                next_line,
                line,
                parts=parts,
                same_baseline_neighbors=same_baseline_neighbor_counts.get(next_index, 0),
            ):
                break
            y_gap = _line_y(next_line) - _line_y(lines[current_index])
            if y_gap <= 0 or y_gap > max(24.0, _line_height(lines[current_index]) * 2.6):
                break
            parts.append(next_text)
            consumed.append(next_index)
            current_index = next_index
        caption = normalize_whitespace(" ".join(parts))
        if _is_table_caption(caption):
            key = caption.casefold()
            if key not in seen:
                seen.add(key)
                candidates.append((caption, line))
                consumed_indices.update(consumed)
                if len(candidates) >= max_candidates:
                    break
    return candidates


def table_caption_candidate_texts(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 250,
) -> List[str]:
    return [text for text, _ in _table_caption_candidate_entries(lines, max_candidates=max_candidates)]


def build_table_caption_candidate_evidence(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 250,
    max_chars: int = 40000,
) -> str:
    parts: List[str] = []
    entries = _table_caption_candidate_entries(lines, max_candidates=max_candidates)
    for text, line in entries:
        entry = (
            f"[{len(parts) + 1}] page={_layout_line_page(line) + 1} "
            f"y={float(line.get('y', 0.0) or 0.0):.1f} x={float(line.get('x', 0.0) or 0.0):.1f} | {text}"
        )
        if sum(len(part) + 1 for part in parts) + len(entry) > max_chars:
            break
        parts.append(entry)
    return "\n".join(parts)


def _looks_like_section_heading(text: str) -> bool:
    s = (text or "").strip().rstrip(":").strip()
    if len(s) < 2 or len(s) > 200:
        return False
    if _FIGURE_CAPTION_START_RE.match(s) or _TABLE_CAPTION_START_RE.match(s):
        return False
    if s.replace(".", "").replace(",", "").strip().isdigit():
        return False
    word_count = len(s.split())
    if word_count < 1 or word_count > 20:
        return False
    return True


def _clean_body_section_headings(value: str) -> List[str]:
    text = normalize_whitespace(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in _BODY_SECTION_EMBEDDED_NUMBER_RE.split(text) if part.strip()]
    out: List[str] = []
    for part in parts or [text]:
        part = part.strip(" ;:-")
        if part:
            out.append(part)
    return out


def _strip_body_section_number_prefix(value: str) -> str:
    match = _BODY_SECTION_NUMBER_PREFIX_RE.match(normalize_whitespace(value))
    return match.group(1).strip() if match else normalize_whitespace(value)


def _looks_like_bio_medrxiv_layout(lines: Sequence[LayoutLine]) -> bool:
    for line in lines[:160]:
        if re.search(r"\b(?:bioRxiv|medRxiv)\s+preprint\b", _layout_text(line), re.IGNORECASE):
            return True
    return False


def _looks_like_reference_title(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < 10 or len(s) > 400:
        return False
    if _FIGURE_CAPTION_START_RE.match(s) or _TABLE_CAPTION_START_RE.match(s):
        return False
    word_count = len(s.split())
    if word_count < 3 or word_count > 60:
        return False
    return True


_REFERENCE_HEADING_RE = re.compile(
    r"^\s*(references|referencias|referências|bibliography|literature cited)\s*[:.]?\s*$",
    re.IGNORECASE,
)
_REFERENCE_ENTRY_START_RE = re.compile(
    r"^\s*(?:"
    r"\[\s*\d+\s*\]|"
    r"\d{1,3}[\.\)]\s+|"
    r"[A-ZÁÉÍÓÚÃÕÇÑ][A-Za-zÀ-ÖØ-öø-ÿ'’`-]{1,40},\s+(?:[A-Z]\.|[A-ZÁÉÍÓÚÃÕÇÑ][A-Za-zÀ-ÖØ-öø-ÿ'’`-]+)"
    r")"
)
_REFERENCE_SECTION_STOP_RE = re.compile(
    r"^\s*(appendix|supplementary\s+material|supporting\s+information|acknowledgements?)\s*[:.]?\s*$",
    re.IGNORECASE,
)


def _reference_start_index(lines: Sequence[LayoutLine]) -> Optional[int]:
    matches = [
        index
        for index, line in enumerate(lines)
        if _REFERENCE_HEADING_RE.match(_layout_text(line))
    ]
    return matches[-1] if matches else None


def _looks_like_reference_entry_start(text: str) -> bool:
    return bool(_REFERENCE_ENTRY_START_RE.match(normalize_whitespace(text)))


def _reference_page_lines_in_reading_order(lines: Sequence[LayoutLine]) -> List[LayoutLine]:
    if len(lines) < 6:
        return sorted(lines, key=lambda line: (_line_y(line), _line_x(line)))
    xs = sorted(float(line.get("x", 0.0) or 0.0) for line in lines)
    gaps = [
        (xs[index + 1] - xs[index], (xs[index] + xs[index + 1]) / 2.0)
        for index in range(len(xs) - 1)
    ]
    if not gaps:
        return sorted(lines, key=lambda line: (_line_y(line), _line_x(line)))
    gap, split_x = max(gaps, key=lambda item: item[0])
    left = [line for line in lines if _line_x(line) <= split_x]
    right = [line for line in lines if _line_x(line) > split_x]
    if gap < 80.0 or len(left) < 3 or len(right) < 3:
        return sorted(lines, key=lambda line: (_line_y(line), _line_x(line)))
    return sorted(left, key=lambda line: (_line_y(line), _line_x(line))) + sorted(
        right,
        key=lambda line: (_line_y(line), _line_x(line)),
    )


def _reference_lines_in_reading_order(lines: Sequence[LayoutLine], start_index: int) -> List[LayoutLine]:
    by_page: Dict[int, List[LayoutLine]] = {}
    for line in lines[start_index + 1 :]:
        text = _layout_text(line)
        if not text or _is_page_marker_text(text):
            continue
        if _REFERENCE_HEADING_RE.match(text):
            continue
        if _REFERENCE_SECTION_STOP_RE.match(text):
            break
        by_page.setdefault(_layout_line_page(line), []).append(line)
    ordered: List[LayoutLine] = []
    for page in sorted(by_page):
        ordered.extend(_reference_page_lines_in_reading_order(by_page[page]))
    return ordered


def _reference_candidate_entries(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 500,
    max_lines_per_reference: int = 10,
    max_chars_per_reference: int = 2200,
) -> List[Tuple[str, LayoutLine]]:
    start_index = _reference_start_index(lines)
    if start_index is None:
        return []

    candidates: List[Tuple[str, LayoutLine]] = []
    current_parts: List[str] = []
    current_line: Optional[LayoutLine] = None
    seen: set[str] = set()

    def flush() -> None:
        nonlocal current_parts, current_line
        if not current_parts or current_line is None:
            current_parts = []
            current_line = None
            return
        text = normalize_whitespace(" ".join(current_parts))
        current_parts = []
        line = current_line
        current_line = None
        if _word_count(text) < 6:
            return
        key = text.casefold()
        if key in seen:
            return
        seen.add(key)
        candidates.append((text, line))

    for line in _reference_lines_in_reading_order(lines, start_index):
        text = _layout_text(line)
        starts_entry = _looks_like_reference_entry_start(text)
        if current_parts and starts_entry:
            flush()
            if len(candidates) >= max_candidates:
                break
        if not current_parts:
            current_line = line
        current_parts.append(text)
        if (
            len(current_parts) >= max_lines_per_reference
            or sum(len(part) + 1 for part in current_parts) >= max_chars_per_reference
        ):
            flush()
            if len(candidates) >= max_candidates:
                break
    flush()
    return candidates[:max_candidates]


def reference_candidate_texts(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 500,
) -> List[str]:
    return [text for text, _ in _reference_candidate_entries(lines, max_candidates=max_candidates)]


def build_reference_candidate_evidence(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 500,
    max_chars: int = 70000,
) -> str:
    parts: List[str] = []
    entries = _reference_candidate_entries(lines, max_candidates=max_candidates)
    for text, line in entries:
        entry = (
            f"[{len(parts) + 1}] page={_layout_line_page(line) + 1} "
            f"y={float(line.get('y', 0.0) or 0.0):.1f} x={float(line.get('x', 0.0) or 0.0):.1f} | {text}"
        )
        if sum(len(part) + 1 for part in parts) + len(entry) > max_chars:
            break
        parts.append(entry)
    return "\n".join(parts)


_BODY_SECTION_KNOWN_RE = re.compile(
    r"^\s*(?:[\dIVXivx]+(?:\.\d+)*[\.\)]?\s+)?"
    r"(?:aims?|introduction|background|overview|results?|discussion|conclusions?|methods?|"
    r"materials\s+and\s+methods|model|models|analysis|analyses|limitations|"
    r"acknowledg(?:e)?ments?|funding|data\s+availability|author\s+contributions?|"
    r"competing\s+interests?|supporting\s+information|supplementary\s+(?:information|material|data)|"
    r"plain\s+language\s+summary)\b",
    re.IGNORECASE,
)
_BODY_SECTION_MAJOR_FULL_RE = re.compile(
    r"^\s*(?:[\dIVXivx]+(?:\.\d+)*[\.\)]?\s+)?"
    r"(?:aims?|introduction|background|overview|results?(?:\s+and\s+discussion)?|discussion|"
    r"conclusions?|methods?|materials\s+and\s+methods|supporting\s+information)\s*:?\s*$",
    re.IGNORECASE,
)
_BODY_SECTION_START_RE = re.compile(
    r"^\s*(?:[\dIVXivx]+(?:\.\d+)*[\.\)]?\s+)?"
    r"(?:introduction|background|rationale|overview)\s*:?\s*$",
    re.IGNORECASE,
)
_BODY_SECTION_NUMBERED_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+){0,3}|[IVX]+)[\.\)]?\s+[A-ZÁÉÍÓÚÃÕÇÑ][^\n]{2,180}$"
)
_BODY_PROCEDURE_ITEM_RE = re.compile(r"^\s*\d+[\.\)]\s+[^:]{2,90}:\s+\S")
_BODY_SECTION_REJECT_RE = re.compile(
    r"\b(?:doi|https?://|copyright|preprint|license|submitted|received|accepted|"
    r"open\s+research\s+europe|last\s+updated|"
    r"correspondence|e-?mail|university|institute|department|laboratory|"
    r"table\s+\d|table\s+s\d|figure\s+\d|figure\s+s\d|fig\.\s*\d|references?|reviewer\s+report|"
    r"reviewer\s+expertise|competing\s+interests?:|grant\s+information|source\s+data)\b",
    re.IGNORECASE,
)
_BODY_PANEL_LABEL_RE = re.compile(r"^\s*[A-Z](?:[,/][A-Z])?[\).:]\s+\S")
_BODY_SECTION_SHORT_REJECT = {
    "yes",
    "no",
    "partly",
    "view",
    "not applicable",
}
_BODY_BACK_MATTER_RE = re.compile(
    r"^\s*(?:figures?\s+(?:and\s+)?figure\s+legends?|figure\s+legends?|"
    r"supplementary\s+figures?|extended\s+data\s+figures?)\s*$",
    re.IGNORECASE,
)
_BODY_STOP_AFTER_HEADING_RE = re.compile(
    r"^\s*(?:acknowledg(?:e)?ments?|draft\s+acknowledg(?:e)?ments?|funding|"
    r"author\s+contributions?|competing\s+interests?|declarations?)\s*:?\s*$",
    re.IGNORECASE,
)
_BODY_BACK_MATTER_HEADING_RE = re.compile(
    r"^\s*(?:acknowledg(?:e)?ments?|draft\s+acknowledg(?:e)?ments?|funding|"
    r"author\s+contributions?|competing\s+interests?)\b",
    re.IGNORECASE,
)
_BODY_AVAILABILITY_HEADING_RE = re.compile(
    r"^\s*(?:(?:data|code|software|materials?|resource|key\s+resources?)\s+availability|"
    r"availability\s+of\s+data|underlying\s+data|extended\s+data|"
    r"supplementary\s+material|supporting\s+information)\b",
    re.IGNORECASE,
)
_BODY_RELAXED_LAYOUT_MAX_PRIMARY_SECTIONS = 8


def _line_font_size(line: LayoutLine) -> float:
    try:
        return float(line.get("font_size", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _median_font_size(lines: Sequence[LayoutLine]) -> float:
    values = sorted(
        _line_font_size(line)
        for line in lines
        if _line_font_size(line) > 0 and _word_count(str(line.get("text", ""))) >= 3
    )
    if not values:
        return 0.0
    return values[len(values) // 2]


def _dominant_body_font_size(lines: Sequence[LayoutLine]) -> float:
    buckets: Dict[float, int] = {}
    for line in lines:
        text = str(line.get("text", "") or "")
        if line.get("bold") or _word_count(text) < 4:
            continue
        font_size = _line_font_size(line)
        if font_size <= 0:
            continue
        bucket = round(font_size, 1)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    if not buckets:
        return _median_font_size(lines)
    return max(buckets.items(), key=lambda item: item[1])[0]


def _line_x(line: LayoutLine) -> float:
    try:
        return float(line.get("x", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _line_y(line: LayoutLine) -> float:
    try:
        return float(line.get("y", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _line_height(line: LayoutLine) -> float:
    try:
        return float(line.get("h", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _dominant_same_column_gap(lines: Sequence[LayoutLine], *, x_tolerance: float = 24.0) -> float:
    gaps: List[float] = []
    by_page: Dict[int, List[LayoutLine]] = {}
    for line in lines:
        if _line_y(line) <= 0 or _line_x(line) <= 0:
            continue
        by_page.setdefault(_layout_line_page(line), []).append(line)
    for page_lines in by_page.values():
        sorted_lines = sorted(page_lines, key=lambda item: (_line_x(item), _line_y(item)))
        for index, line in enumerate(sorted_lines):
            y = _line_y(line)
            x = _line_x(line)
            best_gap = 0.0
            for other in sorted_lines[index + 1 :]:
                if abs(_line_x(other) - x) > x_tolerance:
                    continue
                gap = _line_y(other) - y
                if gap <= 0:
                    continue
                best_gap = gap
                break
            if 6.0 <= best_gap <= 45.0:
                gaps.append(best_gap)
    if not gaps:
        return 0.0
    gaps.sort()
    return gaps[len(gaps) // 2]


def _layout_vertical_gap_map(
    lines: Sequence[LayoutLine],
    *,
    x_tolerance: float = 90.0,
) -> Dict[int, Tuple[float, float]]:
    def is_gap_noise(other: LayoutLine, *, x: float, y: float) -> bool:
        if abs(_line_y(other) - y) < 0.5:
            return True
        text = normalize_whitespace(str(other.get("text", "")))
        return bool(_line_x(other) < x - 10.0 and re.fullmatch(r"\d{1,4}", text))

    by_page: Dict[int, List[Tuple[int, LayoutLine]]] = {}
    for index, line in enumerate(lines):
        by_page.setdefault(_layout_line_page(line), []).append((index, line))

    gaps: Dict[int, Tuple[float, float]] = {}
    for page_lines in by_page.values():
        sorted_lines = sorted(page_lines, key=lambda item: (_line_y(item[1]), _line_x(item[1]), item[0]))
        for position, (index, line) in enumerate(sorted_lines):
            x = _line_x(line)
            y = _line_y(line)
            before = 0.0
            after = 0.0
            for _other_index, other in reversed(sorted_lines[:position]):
                if is_gap_noise(other, x=x, y=y):
                    continue
                if abs(_line_x(other) - x) <= x_tolerance:
                    before = y - _line_y(other)
                    break
            for _other_index, other in sorted_lines[position + 1 :]:
                if is_gap_noise(other, x=x, y=y):
                    continue
                if abs(_line_x(other) - x) <= x_tolerance:
                    after = _line_y(other) - y
                    break
            gaps[index] = (before, after)
    return gaps


def _next_same_column_index(lines: Sequence[LayoutLine], index: int, *, x_tolerance: float = 24.0) -> Optional[int]:
    line = lines[index]
    page = _layout_line_page(line)
    x = _line_x(line)
    y = _line_y(line)
    best_index: Optional[int] = None
    best_y = float("inf")
    for other_index, other in enumerate(lines):
        if other_index == index or _layout_line_page(other) != page:
            continue
        if abs(_line_x(other) - x) > x_tolerance:
            continue
        other_y = _line_y(other)
        if y < other_y < best_y:
            best_y = other_y
            best_index = other_index
    return best_index


def _has_heading_initial(text: str) -> bool:
    if re.match(r"^\s*[∆Δα-ωΑ-Ω]", text or ""):
        return True
    match = re.search(r"[^\W\d_]+", text or "", re.UNICODE)
    if not match:
        return False
    token = match.group(0)
    return bool(token[0].isupper() or any(ch.isupper() for ch in text))


def _looks_like_body_heading_phrase(
    text: str,
    *,
    allow_terminal_period: bool = False,
    allow_terminal_question: bool = False,
) -> bool:
    s = normalize_whitespace(text).strip().strip(":")
    if not s or s.casefold() in _BODY_SECTION_SHORT_REJECT:
        return False
    word_count = _word_count(s)
    if word_count < 1 or word_count > 18:
        return False
    if not _has_heading_initial(s):
        return False
    if s.endswith((",", ";", "!")):
        return False
    if s.endswith("?") and not allow_terminal_question:
        return False
    if s.endswith(".") and not allow_terminal_period:
        return False
    if len(re.findall(r"[.;]", s)) > 1:
        return False
    return True


def _body_heading_candidate_fragment(text: str) -> str:
    s = normalize_whitespace(text).strip()
    if not s:
        return ""
    colon_match = re.match(r"^(.{2,120}?):\s+\S", s)
    if colon_match:
        prefix = colon_match.group(1).strip()
        tail = s[colon_match.end() - 1 :].strip()
        known_match = _BODY_SECTION_KNOWN_RE.match(prefix)
        known_full = bool(known_match and known_match.end() == len(prefix))
        if known_full or _BODY_BACK_MATTER_HEADING_RE.match(prefix):
            if (
                _word_count(s) <= 18
                and _looks_like_body_heading_phrase(tail, allow_terminal_period=True)
                and not tail.rstrip().endswith((".", "?", "!"))
            ):
                return s.rstrip(".").strip()
            return prefix
    period_match = re.match(r"^(.{2,160}?)\.\s+\S", s)
    if period_match:
        prefix = period_match.group(1).strip()
        tail = s[period_match.end() - 1 :].strip()
        if re.fullmatch(r"(?:\d+|phase\s+\d+|evaluation\s+\d+)", prefix, re.IGNORECASE):
            return s
        if _looks_like_body_heading_phrase(prefix, allow_terminal_period=True):
            if (
                _word_count(s) <= 18
                and _word_count(tail) <= 10
                and "," not in tail
                and _looks_like_body_heading_phrase(tail, allow_terminal_period=True)
            ):
                return s.rstrip(".").strip()
            return prefix
    return s.rstrip(".").strip()


def _is_layout_separated_heading_line(
    line: LayoutLine,
    *,
    gap_before: float,
    gap_after: float,
    dominant_gap: float = 0.0,
) -> bool:
    height = max(_line_height(line), 8.0)
    normal_gap = dominant_gap or height * 1.2
    ys = _page_y_scale(line)
    before_ok = gap_before >= max(18.0, normal_gap * 1.6) or (gap_before == 0.0 and _line_y(line) <= 125.0 * ys)
    after_ok = gap_after >= max(8.0, normal_gap * 0.75)
    return bool(before_ok and after_ok)


_BODY_HEADING_CONTINUATION_END_RE = re.compile(
    r"\b(?:a|an|and|are|as|at|be|been|being|between|by|did|do|does|for|from|in|into|is|of|on|or|the|"
    r"through|to|using|via|was|were|with|within|without)\s*$",
    re.IGNORECASE,
)
_BODY_HEADING_CONTINUATION_START_RE = re.compile(r"^\s*(?:and|or)\b", re.IGNORECASE)
_BODY_PROSE_LABEL_RE = re.compile(
    r"^\s*(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+theme\b|"
    r"^\s*the\s+theme\s+of\b",
    re.IGNORECASE,
)


def _is_complete_major_body_heading(text: str) -> bool:
    return bool(_BODY_SECTION_MAJOR_FULL_RE.match(normalize_whitespace(text).strip().strip(":")))


def _looks_like_body_heading_continuation(current_text: str, next_text: str) -> bool:
    current = normalize_whitespace(current_text).strip()
    nxt = normalize_whitespace(next_text).strip()
    if not current or not nxt:
        return False
    current_ends_like_continuation = bool(_BODY_HEADING_CONTINUATION_END_RE.search(current))
    if nxt.endswith((".", "!", "?")) and not (
        current_ends_like_continuation
        and nxt.endswith("?")
        and _word_count(nxt) <= 8
    ):
        return False
    if _FIGURE_CAPTION_START_RE.match(nxt) or _TABLE_CAPTION_START_RE.match(nxt):
        return False
    if _BODY_SECTION_REJECT_RE.search(nxt) or _BODY_PANEL_LABEL_RE.match(nxt):
        return False
    next_words = _word_count(nxt)
    if next_words < 1 or next_words > 10:
        return False
    if current_ends_like_continuation:
        return True
    if _BODY_HEADING_CONTINUATION_START_RE.match(nxt):
        return True
    first_alpha = re.search(r"[^\W\d_]+", nxt, re.UNICODE)
    if first_alpha and first_alpha.group(0)[0].islower() and _word_count(current) >= 4:
        return True
    return False


def _same_baseline_text_neighbor_count_map(
    lines: Sequence[LayoutLine],
    *,
    y_tolerance: float = 1.5,
) -> Dict[int, int]:
    by_page_bucket: Dict[Tuple[int, int], List[int]] = {}
    for index, line in enumerate(lines):
        bucket = int(round(_line_y(line) / y_tolerance))
        by_page_bucket.setdefault((_layout_line_page(line), bucket), []).append(index)

    counts: Dict[int, int] = {}
    for index, line in enumerate(lines):
        page = _layout_line_page(line)
        bucket = int(round(_line_y(line) / y_tolerance))
        x = _line_x(line)
        y = _line_y(line)
        count = 0
        for near_bucket in (bucket - 1, bucket, bucket + 1):
            for other_index in by_page_bucket.get((page, near_bucket), []):
                if other_index == index:
                    continue
                other = lines[other_index]
                if abs(_line_y(other) - y) > y_tolerance:
                    continue
                text = normalize_whitespace(str(other.get("text", "")))
                if not text:
                    continue
                if _line_x(other) < x - 10.0 and re.fullmatch(r"\d{1,4}", text):
                    continue
                count += 1
        counts[index] = count
    return counts


def _is_body_section_candidate_line(
    text: str,
    line: LayoutLine,
    median_font_size: float,
    *,
    gap_before: float = 0.0,
    gap_after: float = 0.0,
    dominant_gap: float = 0.0,
    same_baseline_neighbors: int = 0,
    relaxed_layout: bool = False,
) -> bool:
    s = _body_heading_candidate_fragment(text).strip().strip(":")
    if len(s) < 2 or len(s) > 220:
        return False
    if s.casefold() in _BODY_SECTION_SHORT_REJECT:
        return False
    if _FIGURE_CAPTION_START_RE.match(s) or _TABLE_CAPTION_START_RE.match(s):
        return False
    if ABSTRACT_HEADING_RE.match(s) or KEYWORD_MARKER_RE.search(s):
        return False
    if _BODY_SECTION_REJECT_RE.search(s):
        return False
    if _BODY_BACK_MATTER_HEADING_RE.match(s):
        return False
    if _BODY_PANEL_LABEL_RE.match(s):
        return False
    if _BODY_PROSE_LABEL_RE.match(s):
        return False
    if re.fullmatch(r"[\d\W]+", s):
        return False

    word_count = _word_count(s)
    if word_count < 1 or word_count > 22:
        return False
    bold = bool(line.get("bold"))
    font_size = _line_font_size(line)
    font_too_small = bool(median_font_size and font_size < median_font_size - 1.5)
    prominent = bool(median_font_size and font_size >= median_font_size + 0.5)
    mildly_prominent = bool(median_font_size and font_size >= median_font_size + 0.15)
    layout_separated = _is_layout_separated_heading_line(
        line,
        gap_before=gap_before,
        gap_after=gap_after,
        dominant_gap=dominant_gap,
    )
    section_start_gap = gap_before >= 18.0 or (
        gap_before == 0.0 and _line_y(line) <= 125.0 * _page_y_scale(line)
    )
    known_match = _BODY_SECTION_KNOWN_RE.match(s)
    if known_match:
        tail = s[known_match.end() :].strip()
        if not tail or tail[0] in ":;.-–—(":
            return True
        if _has_heading_initial(tail) and _word_count(tail) <= 16:
            return True
        return False
    if _BODY_PROCEDURE_ITEM_RE.match(s):
        return False
    if _BODY_SECTION_NUMBERED_RE.match(s) and word_count <= 18:
        if font_too_small and not (relaxed_layout and layout_separated):
            return False
        return bool(bold or prominent or layout_separated)
    if same_baseline_neighbors >= 2:
        return False

    if font_too_small and not (
        relaxed_layout
        and layout_separated
        and _looks_like_body_heading_phrase(
            s,
            allow_terminal_period=True,
            allow_terminal_question=True,
        )
    ):
        return False
    if not (bold or prominent):
        if (
            relaxed_layout
            and word_count <= 18
            and layout_separated
            and _looks_like_body_heading_phrase(
                s,
                allow_terminal_period=True,
                allow_terminal_question=True,
            )
        ):
            return True
        if mildly_prominent and _looks_like_body_heading_phrase(s, allow_terminal_question=True):
            return True
        return bool(
            word_count <= 10
            and layout_separated
            and _looks_like_body_heading_phrase(s, allow_terminal_question=True)
        )
    if (
        bold
        and not prominent
        and median_font_size
        and font_size <= median_font_size + 0.15
        and not (layout_separated or section_start_gap)
    ):
        return False
    if not _looks_like_body_heading_phrase(
        s,
        allow_terminal_period=True,
        allow_terminal_question=True,
    ):
        return False
    if word_count > 18:
        return False
    if s.endswith((".", ",")) and not bold:
        return False
    if len(re.findall(r"[.;]", s)) > 2:
        return False
    return True


def _body_start_index(lines: Sequence[LayoutLine]) -> int:
    intro_re = re.compile(
        r"^\s*(?:[\dIVXivx]+(?:\.\d+)*[\.\)]?\s+)?(?:introduction|introdução|introduccion|introducción)\s*:?\s*$",
        re.IGNORECASE,
    )
    for index, line in enumerate(lines):
        if intro_re.match(_layout_text(line)):
            return index
    for index, line in enumerate(lines):
        if _layout_line_page(line) >= 1 and _BODY_SECTION_START_RE.match(_layout_text(line)):
            return index
    return 0


def _join_body_heading_lines(left: str, right: str) -> str:
    left = normalize_whitespace(left)
    right = normalize_whitespace(right)
    if left.endswith("-"):
        return normalize_whitespace(left[:-1] + _body_heading_continuation_fragment(right))
    if re.match(r"^\s*\d+/\d+\.", right):
        return normalize_whitespace(f"{left} {_body_heading_continuation_fragment(right)}")
    return normalize_whitespace(f"{left} {right}")


def _body_heading_continuation_fragment(text: str) -> str:
    text = normalize_whitespace(text)
    if "." not in text:
        return text
    first, _ = text.split(".", 1)
    fragment = first.strip()
    if re.fullmatch(r"\d+/\d+", fragment) or 0 < _word_count(fragment) <= 8:
        return fragment + "."
    return text


def _can_merge_body_heading_line(
    current_text: str,
    current_line: LayoutLine,
    next_text: str,
    next_line: LayoutLine,
    median_font_size: float,
    *,
    next_is_candidate: bool,
) -> bool:
    if _layout_line_page(current_line) != _layout_line_page(next_line):
        return False
    if abs(_line_x(current_line) - _line_x(next_line)) > 24:
        return False
    y_gap = _line_y(next_line) - _line_y(current_line)
    if y_gap <= 0 or y_gap > max(30.0, _line_height(current_line) * 2.8):
        return False
    combined_words = _word_count(current_text) + _word_count(next_text)
    if combined_words > 30:
        return False
    if not current_text.rstrip().endswith("-"):
        if (
            current_line.get("bold")
            and next_line.get("bold")
            and re.match(r"^\s*\d+/\d+\b", normalize_whitespace(next_text))
        ):
            return True
        style_matches = (
            bool(current_line.get("bold")) == bool(next_line.get("bold"))
            and abs(_line_font_size(current_line) - _line_font_size(next_line)) <= 0.2
        )
        if not style_matches:
            return False
        if _is_complete_major_body_heading(current_text):
            return False
        if _looks_like_body_heading_continuation(current_text, next_text):
            return True
        if next_is_candidate:
            return True
        if current_line.get("bold") and next_line.get("bold"):
            return _word_count(next_text) <= 8 and not next_text.rstrip().endswith((".", "?", "!"))
        return False
    font_size = _line_font_size(next_line)
    if (
        median_font_size
        and font_size < median_font_size - 0.2
        and not (current_line.get("bold") and next_line.get("bold"))
    ):
        return False
    fragment = _body_heading_continuation_fragment(next_text)
    return _word_count(fragment) <= 12 and not fragment.rstrip().endswith(("?", "!"))


def _is_page_marker_text(text: str) -> bool:
    return bool(re.fullmatch(r"(?:page\s*)?\d+\s+(?:of|/)\s+\d+|page\s+\d+", text.strip(), re.IGNORECASE))


def _previous_meaningful_text(lines: Sequence[LayoutLine], index: int) -> str:
    for previous in reversed(lines[:index]):
        text = _layout_text(previous)
        if not text or _is_page_marker_text(text):
            continue
        return text
    return ""


def _heading_priority_score(line: LayoutLine, *, median_font_size: float) -> float:
    s = 0.0
    if line.get("bold"):
        s += 2.0
    if line.get("italic"):
        s += 1.0
    fc = (line.get("font_color") or "").lower()
    body_fc = (line.get("doc_body_color") or "").lower()
    if fc and body_fc and fc != body_fc and fc not in {"#000000", "000000", ""}:
        s += 1.5
    ff = (line.get("font_family") or "").lower()
    body_ff = (line.get("doc_body_family") or "").lower()
    if ff and body_ff and ff != body_ff:
        s += 1.0
    ft = (line.get("font_type") or "").lower()
    body_ft = (line.get("doc_body_type") or "").lower()
    if ft and body_ft and ft != body_ft:
        s += 0.5
    fs = float(line.get("font_size", 0.0) or 0.0)
    if median_font_size and fs >= median_font_size + 0.5:
        s += 1.5
    if line.get("is_block_first_line") and 1 <= int(line.get("block_line_count", 0) or 0) <= 3:
        s += 1.0
    return s


def _body_section_candidate_entries_with_index(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 500,
    use_document_spacing: bool = True,
    relaxed_layout: bool = False,
) -> List[Tuple[str, LayoutLine, int, int]]:
    median_font_size = _dominant_body_font_size(lines)
    dominant_gap = _dominant_same_column_gap(lines) if use_document_spacing else 0.0
    start_index = _body_start_index(lines)
    candidates: List[Tuple[str, LayoutLine, int, int]] = []
    seen: set[str] = set()
    in_references = False
    in_back_matter = False
    consumed_indices: set[int] = set()
    gap_map = _layout_vertical_gap_map(lines)
    same_baseline_neighbor_counts = _same_baseline_text_neighbor_count_map(lines)
    for index, line in enumerate(lines):
        if index < start_index or index in consumed_indices:
            continue
        text = _layout_text(line)
        if not text:
            continue
        if re.match(r"^\s*(references|referencias|referências|bibliography|literature cited)\s*[:.]?\s*$", text, re.I):
            in_references = True
        if _BODY_BACK_MATTER_RE.match(text):
            in_back_matter = True
        if _BODY_STOP_AFTER_HEADING_RE.match(text):
            in_back_matter = True
        if in_references or in_back_matter:
            continue
        gap_before, gap_after = gap_map.get(index, (0.0, 0.0))
        if not _is_body_section_candidate_line(
            text,
            line,
            median_font_size,
            gap_before=gap_before,
            gap_after=gap_after,
            dominant_gap=dominant_gap,
            same_baseline_neighbors=same_baseline_neighbor_counts.get(index, 0),
            relaxed_layout=relaxed_layout,
        ):
            continue
        if (
            gap_before == 0.0
            and _line_y(line) <= 125.0
            and not line.get("bold")
            and not _BODY_SECTION_KNOWN_RE.match(text)
            and not _BODY_SECTION_NUMBERED_RE.match(text)
        ):
            previous_text = _previous_meaningful_text(lines, index)
            if previous_text and not previous_text.rstrip().endswith((".", "?", "!", ":", ";")):
                continue
        merged_text = text
        merge_index = index
        merge_indices = [index]
        while True:
            next_index = _next_same_column_index(lines, merge_index)
            if next_index is None or next_index in consumed_indices or next_index < start_index:
                break
            next_line = lines[next_index]
            next_text = _layout_text(next_line)
            next_gap_before, next_gap_after = gap_map.get(next_index, (0.0, 0.0))
            next_is_candidate = _is_body_section_candidate_line(
                next_text,
                next_line,
                median_font_size,
                gap_before=next_gap_before,
                gap_after=next_gap_after,
                dominant_gap=dominant_gap,
                same_baseline_neighbors=same_baseline_neighbor_counts.get(next_index, 0),
                relaxed_layout=relaxed_layout,
            )
            if not _can_merge_body_heading_line(
                merged_text,
                lines[merge_index],
                next_text,
                next_line,
                median_font_size,
                next_is_candidate=next_is_candidate,
            ):
                break
            merged_text = _join_body_heading_lines(merged_text, next_text)
            merge_index = next_index
            merge_indices.append(next_index)
        consumed_indices.update(merge_indices)
        merged_text = _body_heading_candidate_fragment(merged_text)
        if not merged_text:
            continue
        key = merged_text.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates.append((merged_text, line, index, merge_index))
        if len(candidates) >= max_candidates:
            break

    page_count = max((int(ln.get("page", 0) or 0) for ln in lines), default=0) + 1
    supplement_budget = max(0, min(5, page_count // 4))
    if supplement_budget and len(candidates) < max_candidates:
        sup_in_refs = False
        sup_in_back = False
        scored: List[Tuple[float, int, str]] = []
        for index, line in enumerate(lines):
            if index < start_index or index in consumed_indices:
                continue
            text = _layout_text(line)
            if not text:
                continue
            if re.match(
                r"^\s*(references|referencias|referências|bibliography|literature cited)\s*[:.]?\s*$",
                text,
                re.I,
            ):
                sup_in_refs = True
            if _BODY_BACK_MATTER_RE.match(text) or _BODY_STOP_AFTER_HEADING_RE.match(text):
                sup_in_back = True
            if sup_in_refs or sup_in_back:
                continue
            wc = _word_count(text)
            if wc < 2 or wc > 14:
                continue
            if _BODY_SECTION_REJECT_RE.search(text):
                continue
            if _FIGURE_CAPTION_START_RE.match(text) or _TABLE_CAPTION_START_RE.match(text):
                continue
            score = _heading_priority_score(line, median_font_size=median_font_size)
            if score < 4.0:
                continue
            gap_before, _ = gap_map.get(index, (0.0, 0.0))
            if gap_before <= 0:
                continue
            frag = _body_heading_candidate_fragment(text)
            if not frag:
                continue
            key = frag.casefold()
            if key in seen:
                continue
            scored.append((score, index, frag))
        scored.sort(key=lambda item: -item[0])
        admitted = 0
        for score, index, frag in scored:
            if admitted >= supplement_budget:
                break
            key = frag.casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((frag, lines[index], index, index))
            admitted += 1
    return candidates


def _body_section_candidate_entries(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 500,
    use_document_spacing: bool = True,
    relaxed_layout: bool = False,
) -> List[Tuple[str, LayoutLine]]:
    return [
        (text, line)
        for text, line, _start_index, _end_index in _body_section_candidate_entries_with_index(
            lines,
            max_candidates=max_candidates,
            use_document_spacing=use_document_spacing,
            relaxed_layout=relaxed_layout,
        )
    ]


def body_section_candidate_texts(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 500,
    use_document_spacing: bool = True,
    relaxed_layout: bool = False,
) -> List[str]:
    return [
        text
        for text, _ in _body_section_candidate_entries(
            lines,
            max_candidates=max_candidates,
            use_document_spacing=use_document_spacing,
            relaxed_layout=relaxed_layout,
        )
    ]


def _body_section_candidate_context(
    lines: Sequence[LayoutLine],
    index: int,
    candidate_indices: set[int],
    *,
    max_lines: int = 3,
    max_chars: int = 420,
) -> str:
    parts: List[str] = []
    for next_index in range(index + 1, min(len(lines), index + 12)):
        if next_index in candidate_indices:
            break
        text = _layout_text(lines[next_index])
        if not text or _is_page_marker_text(text):
            continue
        if _BODY_STOP_AFTER_HEADING_RE.match(text) or _BODY_BACK_MATTER_RE.match(text):
            break
        if _BODY_PROCEDURE_ITEM_RE.match(text):
            break
        if _FIGURE_CAPTION_START_RE.match(text) or _TABLE_CAPTION_START_RE.match(text):
            break
        parts.append(text)
        if len(parts) >= max_lines or sum(len(part) + 1 for part in parts) >= max_chars:
            break
    return normalize_whitespace(" ".join(parts))[:max_chars]


def build_body_section_candidate_evidence(
    lines: Sequence[LayoutLine],
    *,
    max_candidates: int = 500,
    max_chars: int = 18000,
    include_context: bool = True,
    use_document_spacing: bool = True,
    relaxed_layout: bool = False,
) -> str:
    parts: List[str] = []
    entries = _body_section_candidate_entries_with_index(
        lines,
        max_candidates=max_candidates,
        use_document_spacing=use_document_spacing,
        relaxed_layout=relaxed_layout,
    )
    candidate_indices = {
        index
        for _text, _line, start_index, end_index in entries
        for index in range(start_index, end_index + 1)
    }
    for text, line, start_index, end_index in entries:
        context = _body_section_candidate_context(lines, end_index, candidate_indices) if include_context else ""
        entry = (
            f"[{len(parts) + 1}] page={_layout_line_page(line) + 1} "
            f"y={float(line.get('y', 0.0) or 0.0):.1f} x={float(line.get('x', 0.0) or 0.0):.1f} "
            f"font={_line_font_size(line):.1f} bold={1 if line.get('bold') else 0} | candidate: {text}"
        )
        if context:
            entry = f"{entry}\n    following_text: {context}"
        if sum(len(part) + 1 for part in parts) + len(entry) > max_chars:
            break
        parts.append(entry)
    return "\n".join(parts)


def _content_dedupe_add(target: List[str], items: List[str]) -> None:
    """Like _dedupe_add but also drops substring/near-duplicate matches (>=90% fuzzy)."""
    try:
        from rapidfuzz.fuzz import token_set_ratio
        have_fuzzy = True
    except ImportError:
        have_fuzzy = False

    norm_existing = [" ".join(x.split()).lower() for x in target]
    for it in items:
        k = " ".join(str(it).split()).lower()
        if not k:
            continue
        if k in norm_existing:
            continue
        if any(k != n and (k in n or n in k) for n in norm_existing):
            continue
        if have_fuzzy and any(token_set_ratio(k, n) >= 90 for n in norm_existing):
            continue
        target.append(it)
        norm_existing.append(k)


_TABLE_CAPTION_LABEL_RE = re.compile(
    r"^\s*((?:table|tabla|tabela|tab)\s*\.?\s*(?:s\s*)?(?:\d+[A-Za-z]?|[IVX]+))",
    re.IGNORECASE,
)


def _table_caption_label_key(text: str) -> str:
    m = _TABLE_CAPTION_LABEL_RE.match(text or "")
    if not m:
        return ""
    return re.sub(r"\s+", "", m.group(1).casefold().replace(".", ""))


def _table_caption_dedupe_add(target: List[str], items: List[str]) -> None:
    try:
        from rapidfuzz.fuzz import token_set_ratio
        have_fuzzy = True
    except ImportError:
        have_fuzzy = False

    for it in items:
        candidate = str(it).strip()
        k = " ".join(candidate.split()).lower()
        if not k:
            continue
        label = _table_caption_label_key(k)
        duplicate = False
        for existing in target:
            existing_norm = " ".join(existing.split()).lower()
            existing_label = _table_caption_label_key(existing_norm)
            if label and existing_label and label != existing_label:
                continue
            if k == existing_norm or (k != existing_norm and (k in existing_norm or existing_norm in k)):
                duplicate = True
                break
            if have_fuzzy and token_set_ratio(k, existing_norm) >= 90:
                duplicate = True
                break
        if not duplicate:
            target.append(candidate)


def predict_content_fields_from_alto(
    lines: Sequence[LayoutLine],
    chat: Callable[..., str],
    max_chars: int = 25000,  # pylint: disable=unused-argument
    max_tokens: int = 2000,  # pylint: disable=unused-argument
    body_sections_max_tokens: int = 2500,
    references_max_chars: int = 40000,
    references_max_tokens: int = 4500,
    tables_figures_max_chars: int = 110000,  # pylint: disable=unused-argument
    tables_figures_max_tokens: int = 3500,
) -> MetadataRecord:
    strip_body_numbers = _looks_like_bio_medrxiv_layout(lines)
    content_lines = prune_layout_lines(lines)
    all_text = "\n".join(line["text"] for line in content_lines if line.get("text"))
    body_candidate_values = body_section_candidate_texts(content_lines)
    body_candidate_text = build_body_section_candidate_evidence(content_lines)
    legacy_body_candidate_values: List[str] = []
    legacy_body_candidate_text = ""
    relaxed_body_candidate_values: List[str] = []
    relaxed_body_candidate_text = ""
    if strip_body_numbers:
        legacy_body_candidate_values = body_section_candidate_texts(content_lines, use_document_spacing=False)
        legacy_body_candidate_text = build_body_section_candidate_evidence(
            content_lines,
            include_context=False,
            use_document_spacing=False,
        )
        relaxed_body_candidate_values = body_section_candidate_texts(
            content_lines,
            use_document_spacing=False,
            relaxed_layout=True,
        )
        if len(relaxed_body_candidate_values) > len(legacy_body_candidate_values) + 3:
            relaxed_body_candidate_text = build_body_section_candidate_evidence(
                content_lines,
                include_context=False,
                use_document_spacing=False,
                relaxed_layout=True,
            )
    all_body_candidate_values = body_candidate_values + legacy_body_candidate_values + relaxed_body_candidate_values
    figure_candidate_values = figure_caption_candidate_texts(content_lines)
    figure_candidate_text = build_figure_caption_candidate_evidence(content_lines)
    table_candidate_values = table_caption_candidate_texts(content_lines)
    table_candidate_text = build_table_caption_candidate_evidence(content_lines)
    ref_start_idx = _reference_start_index(content_lines)
    if ref_start_idx is not None and _estimate_column_count(content_lines[ref_start_idx + 1:]) >= 2:
        ref_content_lines = sorted(
            content_lines,
            key=lambda item: (
                int(item.get("page", 0) or 0),
                float(item.get("y", 0.0) or 0.0),
                float(item.get("x", 0.0) or 0.0),
            ),
        )
    else:
        ref_content_lines = content_lines
    reference_candidate_values = reference_candidate_texts(ref_content_lines)
    reference_candidate_text = build_reference_candidate_evidence(
        ref_content_lines,
        max_chars=references_max_chars,
    )
    empty: MetadataRecord = {
        "body_sections": [],
        "figure_captions": [],
        "table_captions": [],
        "reference_titles": [],
        "reference_dois": [],
    }
    if not all_text.strip():
        return empty

    body_sections: List[str] = []
    figure_captions: List[str] = []
    table_captions: List[str] = []
    reference_titles: List[str] = []
    reference_dois: List[str] = []

    def _dedupe_add(target: List[str], items: List[str]) -> None:
        _content_dedupe_add(target, items)

    def _dedupe_table_add(target: List[str], items: List[str]) -> None:
        _table_caption_dedupe_add(target, items)

    def _call(system_prompt: str, user_text: str, out_tokens: int, step_name: str) -> Optional[Dict[str, Any]]:
        try:
            raw = chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_tokens=out_tokens,
                step_name=step_name,
            )
        except Exception:
            return None
        try:
            return extract_json_from_text(raw)
        except Exception:
            return None

    def _filter(items: List[str], pred: Callable[[str], bool]) -> List[str]:
        return [str(v).strip() for v in items if isinstance(v, str) and pred(str(v))]

    def _add_body_payload(payload: Optional[Dict[str, Any]]) -> None:
        if not payload:
            return
        values = payload.get("body_sections") or []
        if not isinstance(values, list):
            return
        cleaned_values: List[str] = []
        for value in values:
            if isinstance(value, str):
                cleaned_values.extend(_clean_body_section_headings(value))
        if strip_body_numbers:
            cleaned_values = [_strip_body_section_number_prefix(value) for value in cleaned_values]
        supported_values = [
            require_extractive_support(str(value), all_body_candidate_values)
            for value in cleaned_values
        ]
        _dedupe_add(
            body_sections,
            _filter(
                supported_values,
                lambda value: _looks_like_section_heading(value)
                and (strip_body_numbers or not _BODY_AVAILABILITY_HEADING_RE.match(value)),
            ),
        )

    # Run the passes concurrently; merges after are serial and so stay thread-safe.
    with ThreadPoolExecutor(max_workers=6) as _content_ex:
        body_fut = (
            _content_ex.submit(
                _call,
                BODY_SECTIONS_EXTRACTION_PROMPT,
                body_candidate_text,
                body_sections_max_tokens,
                "CONTENT_BODY_SECTIONS",
            )
            if len(body_candidate_text.strip()) > 100 else None
        )
        legacy_body_fut = (
            _content_ex.submit(
                _call,
                BODY_SECTIONS_EXTRACTION_PROMPT,
                legacy_body_candidate_text,
                body_sections_max_tokens,
                "CONTENT_BODY_SECTIONS",
            )
            if len(legacy_body_candidate_text.strip()) > 100 else None
        )
        figure_fut = (
            _content_ex.submit(
                _call,
                FIGURE_CAPTIONS_SELECTION_PROMPT,
                figure_candidate_text,
                tables_figures_max_tokens,
                "CONTENT_FIGURE_CAPTIONS",
            )
            if len(figure_candidate_text.strip()) > 20 else None
        )
        table_fut = (
            _content_ex.submit(
                _call,
                TABLE_CAPTIONS_SELECTION_PROMPT,
                table_candidate_text,
                tables_figures_max_tokens,
                "CONTENT_TABLE_CAPTIONS",
            )
            if len(table_candidate_text.strip()) > 20 else None
        )
        refs_fut = (
            _content_ex.submit(
                _call,
                REFERENCES_EXTRACTION_PROMPT,
                reference_candidate_text,
                references_max_tokens,
                "CONTENT_REFERENCES",
            )
            if len(reference_candidate_text.strip()) > 20 else None
        )
        body_payload = body_fut.result() if body_fut is not None else None
        legacy_body_payload = legacy_body_fut.result() if legacy_body_fut is not None else None
        figure_payload = figure_fut.result() if figure_fut is not None else None
        table_payload = table_fut.result() if table_fut is not None else None
        refs_payload = refs_fut.result() if refs_fut is not None else None

    _add_body_payload(body_payload)
    _add_body_payload(legacy_body_payload)
    if (
        strip_body_numbers
        and len(body_sections) <= _BODY_RELAXED_LAYOUT_MAX_PRIMARY_SECTIONS
        and len(relaxed_body_candidate_text.strip()) > 100
    ):
        _add_body_payload(
            _call(
                BODY_SECTIONS_EXTRACTION_PROMPT,
                relaxed_body_candidate_text,
                body_sections_max_tokens,
                "CONTENT_BODY_SECTIONS",
            )
        )

    if figure_payload:
        figures_list = figure_payload.get("figures") or figure_payload.get("figure_captions") or []
        if isinstance(figures_list, list):
            supported_figures = [
                require_extractive_support(str(value), figure_candidate_values)
                for value in figures_list
                if isinstance(value, str)
            ]
            _dedupe_add(figure_captions, _filter(supported_figures, _is_figure_caption))

    if table_payload:
        tables_list = table_payload.get("tables") or table_payload.get("table_captions") or []
        if isinstance(tables_list, list):
            supported_tables = [
                require_extractive_support(str(value), table_candidate_values)
                for value in tables_list
                if isinstance(value, str)
            ]
            _dedupe_table_add(table_captions, _filter(supported_tables, _is_table_caption))

    if refs_payload:
        refs = refs_payload.get("references") or []
        if isinstance(refs, list):
            new_titles: List[str] = []
            new_dois: List[str] = []
            for r in refs:
                if not isinstance(r, dict):
                    continue
                title = str(r.get("title") or "").strip()
                if title and _looks_like_reference_title(title):
                    new_titles.append(title)
                doi_candidate = str(r.get("doi") or "").strip()
                if doi_candidate:
                    m = _CONTENT_DOI_RE.search(doi_candidate)
                    if m:
                        doi = m.group(0).lower().rstrip(".,;)")
                        if require_extractive_support(doi, reference_candidate_values):
                            new_dois.append(doi)
            supported_titles = [
                require_extractive_support(title, reference_candidate_values)
                for title in new_titles
            ]
            _dedupe_add(reference_titles, _filter(supported_titles, _looks_like_reference_title))
            _dedupe_add(reference_dois, new_dois)

    return {
        "body_sections": body_sections,
        "figure_captions": figure_captions,
        "table_captions": table_captions,
        "reference_titles": reference_titles,
        "reference_dois": reference_dois,
    }


_CONTENT_FIELD_PREDICATES: Dict[str, Callable[[str], bool]] = {
    "body_sections": _looks_like_section_heading,
    "figure_captions": _is_figure_caption,
    "table_captions": _is_table_caption,
    "reference_titles": _looks_like_reference_title,
}


_ALTO_PREFERRED_CONTENT_FIELDS = {
    "body_sections",
    "figure_captions",
    "table_captions",
    "reference_titles",
    "reference_dois",
}


_REFERENCE_UNION_FIELDS = {"reference_dois", "reference_titles"}


def merge_content_fields(tei_content: MetadataRecord, llm_content: MetadataRecord) -> MetadataRecord:
    """Prefer ALTO/LLM content_fields; fall back to TEI when the LLM path returned nothing supported.

    Reference fields (reference_dois, reference_titles) are unioned rather than
    replaced — GROBID's structural extractor often finds DOIs the LLM stage
    later drops, and the previous "LLM if non-empty else TEI" rule was costing
    340 correctly-extracted reference DOIs across a 149-doc smoke (worst case
    scielo_br/S0101-28002026000200403: TEI 34, LLM 1, dropped 33 of 33 gold).
    """
    out: MetadataRecord = dict(tei_content)
    for key in ("body_sections", "figure_captions", "table_captions", "reference_titles", "reference_dois"):
        predicate = _CONTENT_FIELD_PREDICATES.get(key)
        tei_items = [str(i) for i in (tei_content.get(key) or []) if isinstance(i, str)]
        llm_items = [str(i) for i in (llm_content.get(key) or []) if isinstance(i, str)]
        if predicate is not None:
            tei_items = [i for i in tei_items if predicate(i)]
            llm_items = [i for i in llm_items if predicate(i)]
        if key in _REFERENCE_UNION_FIELDS:
            chosen = tei_items + llm_items
        elif key in _ALTO_PREFERRED_CONTENT_FIELDS:
            chosen = llm_items if llm_items else tei_items
        else:
            chosen = tei_items if tei_items else llm_items
        merged: List[str] = []
        seen: Dict[str, None] = {}
        for item in chosen:
            k = " ".join(item.split()).lower()
            if k and k not in seen:
                merged.append(item)
                seen[k] = None
        out[key] = merged
    return out


def enrich_references_with_crossref(
    pred: MetadataRecord,
    tei_path: Path,
    crossref_client: Any = None,
    max_lookups: int = 80,
    max_workers: int = 5,
) -> MetadataRecord:
    """Augment empty reference fields by looking up GROBID biblStructs via Crossref.

    All-or-nothing gate: only fires when pred has zero references. Per-bibl
    enrichment was tried (commit 9ed6941) and refuted in CI — adding Crossref
    DOIs on top of an existing bibliography inflated false positives faster
    than it recovered missing DOIs (biorxiv reference_f1 0.908 -> 0.754 due to
    +137 spurious DOIs). The CrossrefClient's MIN_TITLE_JACCARD floor wasn't
    strict enough to prevent generic-title collisions on bioRxiv preprints.
    """
    import xml.etree.ElementTree as _ET

    from .crossref import CrossrefClient as _CrossrefClient

    if pred.get("reference_titles") or pred.get("reference_dois"):
        return pred

    if crossref_client is None:
        crossref_client = _CrossrefClient()

    try:
        tree = _ET.parse(tei_path)
    except Exception:
        return pred

    out = dict(pred)
    out["reference_dois"] = list(pred.get("reference_dois") or [])
    out["reference_titles"] = list(pred.get("reference_titles") or [])

    def _strip(tag: str) -> str:
        return tag.split("}", 1)[1] if "}" in tag else tag

    def _text(el: Any) -> str:
        return " ".join(el.itertext()).strip() if el is not None else ""

    signatures: List[Dict[str, Any]] = []
    for bibl in tree.getroot().iter():
        if _strip(bibl.tag) != "biblStruct":
            continue
        has_doi = False
        for inner in bibl.iter():
            if _strip(inner.tag) == "idno" and (inner.get("type") or "").lower() == "doi" and _text(inner):
                has_doi = True
                break
        if has_doi:
            continue

        title = ""
        for inner in bibl.iter():
            if _strip(inner.tag) == "title" and (inner.get("level") or "").lower() == "a":
                title = _text(inner)
                if title:
                    break
        if not title:
            for inner in bibl.iter():
                if _strip(inner.tag) == "title":
                    title = _text(inner)
                    if title:
                        break
        if not title or len(title) < 15:
            continue

        authors: List[str] = []
        for pers in bibl.iter():
            if _strip(pers.tag) != "persName":
                continue
            surname = ""
            for ch in pers:
                if _strip(ch.tag) == "surname":
                    surname = _text(ch)
            if surname:
                authors.append(surname)
            if len(authors) >= 4:
                break

        journal = ""
        for inner in bibl.iter():
            if _strip(inner.tag) == "title" and (inner.get("level") or "").lower() in {"j", "m"}:
                journal = _text(inner)
                if journal:
                    break

        year = ""
        for inner in bibl.iter():
            if _strip(inner.tag) == "date":
                year = (inner.get("when") or _text(inner) or "").strip()
                m = re.search(r"\b(19|20)\d{2}\b", year)
                if m:
                    year = m.group(0)
                if year:
                    break

        signatures.append({"title": title, "authors": authors, "year": year, "journal": journal})
        if len(signatures) >= max_lookups:
            break

    if not signatures:
        return out

    def _lookup(sig: Dict[str, Any]) -> Dict[str, str]:
        result: Dict[str, str] = crossref_client.lookup(
            title=sig["title"], authors=sig["authors"], year=sig["year"], journal=sig["journal"]
        )
        return result

    recovered_dois: List[str] = []
    recovered_titles: List[str] = []
    if max_workers <= 1:
        for sig in signatures:
            hit = _lookup(sig)
            if hit.get("doi"):
                recovered_dois.append(hit["doi"])
            if hit.get("title"):
                recovered_titles.append(hit["title"])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for hit in ex.map(_lookup, signatures):
                if hit.get("doi"):
                    recovered_dois.append(hit["doi"])
                if hit.get("title"):
                    recovered_titles.append(hit["title"])

    existing_dois = set(out["reference_dois"])
    for d in recovered_dois:
        if d not in existing_dois:
            out["reference_dois"].append(d)
            existing_dois.add(d)

    existing_title_keys = {" ".join(t.split()).lower() for t in out["reference_titles"]}
    for t in recovered_titles:
        key = " ".join(t.split()).lower()
        if key and key not in existing_title_keys:
            out["reference_titles"].append(t)
            existing_title_keys.add(key)
    return out


def build_prediction(
    context: DocumentContext,
    chat: Callable[..., str],
    per_document_llm_workers: int,
) -> MetadataRecord:
    with get_tracer().start_as_current_span("document_prediction") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("document.id", context.record_id)
        return _build_prediction_inner(context, chat, per_document_llm_workers)


def _build_prediction_inner(
    context: DocumentContext,
    chat: Callable[..., str],
    per_document_llm_workers: int,
) -> MetadataRecord:
    tasks: Dict[str, Callable[[], Any]] = {
        "header_metadata": lambda: predict_header_metadata(context, chat),
        "tei_metadata_pair": lambda: _predict_tei_metadata_with_validation(context, chat),
        "abstract_from_candidates": lambda: select_abstract_from_candidates(context, chat),
        # OCR_CLEANUP was a warm-up call whose only consumer was the abstract
        # extraction below; we now feed raw OCR text straight into
        # extract_abstract_from_ocr, dropping one LLM call per doc.
        "ocr_abstract": lambda: extract_abstract_from_ocr(build_ocr_input(context.lines)[:8000], chat),
    }

    def _default(name: str) -> Any:
        if name == "tei_metadata_pair":
            return ({}, {})
        return "" if name in {"abstract_from_candidates", "ocr_abstract"} else {}

    results: Dict[str, Any] = {}
    worker_count = max(1, int(per_document_llm_workers))
    if worker_count == 1:
        for name, task in tasks.items():
            try:
                results[name] = task()
            except Exception:
                results[name] = _default(name)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(with_otel_context(task)): name for name, task in tasks.items()}
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    results[name] = future.result()
                except Exception:
                    results[name] = _default(name)

    header_metadata = normalize_metadata(results.get("header_metadata") or {})
    tei_metadata_pair: Tuple[Any, Any] = results.get("tei_metadata_pair") or ({}, {})
    tei_metadata = normalize_metadata(tei_metadata_pair[0] or {})
    validated_tei_metadata = normalize_metadata(tei_metadata_pair[1] or {})
    ocr_abstract = str(results.get("ocr_abstract") or "")

    # Align with the original exp30 pipeline: start from TEI-extracted fields only,
    # and use LLM TEI outputs strictly as abstract candidates (not as field sources).
    metadata = normalize_metadata(context.tei_fields)

    preferred_language = canonical_language_code(str(metadata.get("language", "")))
    if preferred_language == "unknown":
        abstract_text = str(metadata.get("abstract", ""))
        if abstract_text and not is_mixed_language(abstract_text):
            preferred_language = detect_language(abstract_text)
    if preferred_language == "unknown":
        preferred_language = detect_language(metadata.get("title", ""))

    abstract = choose_abstract_candidate_from_sources(
        [
            ("tei_fields", metadata.get("abstract", "")),
            ("header_metadata", header_metadata.get("abstract", "")),
            ("tei_metadata", tei_metadata.get("abstract", "")),
            ("tei_validated_metadata", validated_tei_metadata.get("abstract", "")),
            *build_abstract_candidates(context),
            ("abstract_from_candidates", str(results.get("abstract_from_candidates") or "")),
            ("ocr_abstract", ocr_abstract),
        ],
        preferred_language,
    )
    if abstract:
        metadata["abstract"] = abstract

    if header_metadata.get("title"):
        preferred_title_language = canonical_language_code(metadata.get("language", ""))
        metadata["title"] = choose_title_candidate(header_metadata["title"], preferred_title_language)
    if header_metadata.get("authors"):
        metadata["authors"] = header_metadata["authors"]
    if header_metadata.get("affiliations"):
        metadata["affiliations"] = header_metadata["affiliations"]
    keyword_candidate_sets = [
        *extract_keyword_candidate_sets_from_front_matter(
            context,
            chat,
            title=metadata.get("title", ""),
            abstract=metadata.get("abstract", ""),
        ),
        ("tei", normalize_keyword_values(metadata.get("keywords") or [])),
        ("header_llm", normalize_keyword_values(header_metadata.get("keywords") or [])),
        ("tei_llm", normalize_keyword_values(tei_metadata.get("keywords") or [])),
        ("tei_validated_llm", normalize_keyword_values(validated_tei_metadata.get("keywords") or [])),
    ]
    keyword_language = canonical_language_code(metadata.get("language", ""))
    if keyword_language == "unknown":
        keyword_language = preferred_language
    selected_keywords = select_keywords_from_candidates(
        context,
        chat,
        keyword_candidate_sets,
        title=metadata.get("title", ""),
        abstract=metadata.get("abstract", ""),
        preferred_language=keyword_language,
    )
    if (
        not selected_keywords
        and not _keyword_marker_evidence_text(context)
        and not any(values for _, values in keyword_candidate_sets)
    ):
        selected_keywords = infer_keywords_from_metadata(
            context,
            chat,
            title=metadata.get("title", ""),
            abstract=metadata.get("abstract", ""),
        )
    metadata["keywords"] = selected_keywords

    identifier_candidate_sets = [
        ("tei", metadata.get("identifiers") or []),
        ("tei_llm", tei_metadata.get("identifiers") or []),
        ("tei_validated_llm", validated_tei_metadata.get("identifiers") or []),
        ("record_id", scielo_identifiers_from_record_id(context.record_id)),
    ]
    metadata["identifiers"] = select_article_identifiers_from_candidates(
        context,
        chat,
        identifier_candidate_sets,
        title=metadata.get("title", ""),
    )
    return normalize_metadata(metadata)


def process_record(
    row: ManifestRow,
    settings: PipelineSettings,
    chat: Callable[..., str],
) -> Dict[str, Any]:
    paths = build_document_paths(row, settings.output_dir, parser=settings.parser)
    run_grobid(
        paths.pdf_path,
        paths.tei_path,
        grobid_url=settings.grobid_url,
        parser=settings.parser,
    )
    run_pdfalto(
        paths.pdf_path,
        paths.alto_path,
        pdfalto_bin=settings.pdfalto_bin,
        start_page=settings.pdfalto_start_page,
        end_page=settings.pdfalto_end_page,
    )

    if paths.prediction_path.exists() and not settings.rerun:
        prediction = json.loads(paths.prediction_path.read_text(encoding="utf-8"))
    else:
        context = build_document_context(paths)
        prediction = build_prediction(context, chat, settings.per_document_llm_workers)
        paths.prediction_path.write_text(json.dumps(prediction, ensure_ascii=True, indent=2), encoding="utf-8")

    gold = extract_oai_dc(paths.xml_path)
    metrics = evaluate_record(prediction, gold)
    return {
        "record_id": row["record_id"],
        "pred": prediction,
        "gold": gold,
        "metrics": metrics,
    }


def run_pipeline(settings: PipelineSettings) -> Dict[str, Any]:
    init_telemetry()
    ensure_dir(settings.output_dir / "tei" / settings.parser)
    ensure_dir(settings.output_dir / "alto")
    ensure_dir(settings.output_dir / "predictions" / settings.parser)

    manifest_path = settings.manifest_path
    if manifest_path.suffix.lower() in {".parquet", ".pq"}:
        manifest = load_parquet_manifest(manifest_path, settings.output_dir)
    else:
        manifest = load_manifest(manifest_path)
    if settings.limit is not None:
        manifest = manifest[: settings.limit]

    if settings.openai_api_key or settings.openai_model:
        if not settings.openai_api_key or not settings.openai_model:
            raise ValueError("Both openai_api_key and openai_model are required when using OpenAI API.")
        client: Any = OpenAIClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
        )
    else:
        client = AoaiPool(settings.pool_path, routing=settings.llm_pool_routing)
    semaphore = threading.Semaphore(max(1, int(settings.llm_concurrency)))

    def chat(
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
        *,
        step_name: str = "",
    ) -> str:
        with semaphore:
            return str(client.chat(messages, temperature=temperature, max_tokens=max_tokens, step_name=step_name))

    per_document: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    worker_count = settings.workers or max(1, min(4, os.cpu_count() or 4))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(process_record, row, settings, chat): row["record_id"] for row in manifest}
        for future in as_completed(future_map):
            record_id = future_map[future]
            try:
                per_document.append(future.result())
            except Exception as error:
                errors.append({"record_id": record_id, "error": str(error)})

    summary = aggregate_metrics(per_document)
    (settings.output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    with (settings.output_dir / "per_document.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_document:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    write_root_cause_report(
        per_document,
        summary,
        settings.output_dir / "root_causes.md",
        "Metadata Enrichment Pipeline",
    )

    if errors:
        (settings.output_dir / "errors.json").write_text(
            json.dumps(errors, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    return {"summary": summary, "errors": errors, "per_document": per_document}
