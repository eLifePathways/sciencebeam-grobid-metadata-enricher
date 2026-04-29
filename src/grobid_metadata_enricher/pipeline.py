# pylint: disable=too-many-lines
from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    CONTENT_EXTRACTION_PROMPT,
    HEADER_METADATA_PROMPT,
    KEYWORD_TRANSLATION_PROMPT,
    OCR_CLEANUP_PROMPT,
    REFERENCES_EXTRACTION_PROMPT,
    TABLES_FIGURES_EXTRACTION_PROMPT,
    TEI_METADATA_PROMPT,
)
from .telemetry import get_tracer, init_telemetry, with_otel_context

DISCLAIMER_RE = re.compile(
    r"(preprint|scielo|deposit|submitted|presentado|condi[cç]iones|condi[cç][aã]o|declaram|responsab)",
    re.IGNORECASE,
)
ABSTRACT_MARKER_RE = re.compile(r"\b(abstract|resumo|resumen)\b", re.IGNORECASE)
ENGLISH_MARKER_RE = re.compile(r"\babstract\b", re.IGNORECASE)
PORTUGUESE_MARKER_RE = re.compile(r"\bresumo\b", re.IGNORECASE)
SPANISH_MARKER_RE = re.compile(r"\bresumen\b", re.IGNORECASE)
ENGLISH_START_RE = re.compile(r"^\s*(abstract|the|this|we|in|a|an)\b", re.IGNORECASE)
ROMANCE_START_RE = re.compile(r"^\s*(resumo|resumen)[:\s]", re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-ZáéíóúãõçñÁÉÍÓÚÃÕÇÑ]+")
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
    indices = [index for index, line in enumerate(lines) if ABSTRACT_MARKER_RE.search(line.get("text", ""))]
    blocks: List[str] = []
    if indices:
        for index in indices[:max_blocks]:
            start = max(0, index - prefix_lines)
            end = min(len(lines), index + suffix_lines)
            text = " ".join(line["text"] for line in lines[start:end])
            blocks.append(normalize_whitespace(text))
    else:
        text = " ".join(line["text"] for line in lines[:fallback_lines])
        blocks.append(normalize_whitespace(text))
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


def dedupe_blocks(blocks: Sequence[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for text in blocks:
        key = normalize_whitespace(text).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique


def build_abstract_candidates(context: DocumentContext) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    for text in context.tei_abstracts:
        if not DISCLAIMER_RE.search(text):
            blocks.append(("tei_abstract", text))
    for index, block in enumerate(
        marker_windows(
            context.lines,
            max_blocks=3,
            prefix_lines=2,
            suffix_lines=14,
            fallback_lines=120,
        ),
        start=1,
    ):
        blocks.append((f"alto_block_{index}", block))
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
        if not DISCLAIMER_RE.search(text):
            blocks.append(text)
    for block in marker_windows(
        context.lines,
        max_blocks=4,
        prefix_lines=0,
        suffix_lines=18,
        fallback_lines=160,
    ):
        if not DISCLAIMER_RE.search(block):
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
    if DISCLAIMER_RE.search(text):
        score -= 5.0
    if ENGLISH_START_RE.search(text):
        score += 2.0
    if ROMANCE_START_RE.search(text):
        score -= 1.0
    if 50 <= word_count <= 300:
        score += 1.0
    if word_count < 30:
        score -= 2.0
    if word_count > 500:
        score -= 1.0
    return score


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
    if any(not DISCLAIMER_RE.search(candidate) for candidate in available):
        available = [candidate for candidate in available if not DISCLAIMER_RE.search(candidate)]
    if preferred_language and preferred_language != "unknown":
        matching = [candidate for candidate in available if detect_language(candidate) == preferred_language]
        if matching:
            return max(matching, key=score_abstract_candidate)
    return max(available, key=score_abstract_candidate)


def add_scielo_identifiers(record_id: str, identifiers: Sequence[str]) -> List[str]:
    values = [value for value in identifiers if value]
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


def predict_header_metadata(context: DocumentContext, chat: Callable[..., str]) -> MetadataRecord:
    lines = (context.first_page_lines or context.lines)[:80]
    messages = [
        {"role": "system", "content": HEADER_METADATA_PROMPT},
        {"role": "user", "content": format_header_lines(lines)},
    ]
    raw = chat(messages, temperature=0.0, max_tokens=700, step_name="HEADER_METADATA")
    return normalize_metadata(safe_extract_json(raw))


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
    return str(payload.get("abstract", "")).strip()


def clean_ocr_text(context: DocumentContext, chat: Callable[..., str]) -> str:
    messages = [
        {"role": "system", "content": OCR_CLEANUP_PROMPT},
        {"role": "user", "content": build_ocr_input(context.lines)[:8000]},
    ]
    return normalize_whitespace(chat(messages, temperature=0.0, max_tokens=800, step_name="OCR_CLEANUP"))


def extract_abstract_from_ocr(clean_text: str, chat: Callable[..., str]) -> str:
    messages = [
        {"role": "system", "content": ABSTRACT_EXTRACTION_PROMPT},
        {"role": "user", "content": slice_near_abstract_marker(clean_text)},
    ]
    payload = safe_extract_json(chat(messages, temperature=0.0, max_tokens=600, step_name="ABSTRACT_FROM_OCR"))
    return str(payload.get("abstract", "")).strip()


_CONTENT_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s<>\"'\\)]+", re.IGNORECASE)


def predict_content_fields_from_alto(
    lines: Sequence[LayoutLine],
    chat: Callable[..., str],
    max_chars: int = 25000,
    max_tokens: int = 2000,
    references_max_chars: int = 40000,
    references_max_tokens: int = 4500,
    tables_figures_max_chars: int = 110000,
    tables_figures_max_tokens: int = 3500,
) -> MetadataRecord:
    """Extract body/figure/table/reference content from full-paper ALTO text.

    Runs three LLM passes so long papers do not lose content past an input cap:
      head pass over the first max_chars (body sections, inline figures/tables,
      references visible in the head);
      references pass over the last references_max_chars (reference list at the
      tail of the paper, with a larger output budget so long bibliographies are
      not truncated);
      tables/figures pass over the first tables_figures_max_chars (tables and
      figures scattered through the body; a wider window than the head pass so
      captions past page 20 or so are still covered).

    Each pass appends into shared per-key buckets with whitespace-normalised
    case-insensitive dedup. Callers typically merge the result with the TEI
    content_fields via merge_content_fields.
    """
    all_text = "\n".join(line["text"] for line in lines if line.get("text"))
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
        seen = {" ".join(x.split()).lower() for x in target}
        for it in items:
            k = " ".join(str(it).split()).lower()
            if k and k not in seen:
                target.append(it)
                seen.add(k)

    head_text = all_text[:max_chars]
    tail_text = all_text[-references_max_chars:]
    # Skip the first max_chars of the tables/figures pass input because
    # CONTENT_HEAD already reads that region; keeping them in would double-bill
    # a ~1-2k-prompt-token overlap per document without adding new coverage.
    full_for_tf = all_text[max_chars:tables_figures_max_chars]

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

    # Run the 3 passes concurrently; merges after are serial and so stay thread-safe.
    with ThreadPoolExecutor(max_workers=3) as _content_ex:
        head_fut = _content_ex.submit(_call, CONTENT_EXTRACTION_PROMPT, head_text, max_tokens, "CONTENT_HEAD")
        refs_fut = (
            _content_ex.submit(
                _call, REFERENCES_EXTRACTION_PROMPT, tail_text, references_max_tokens, "CONTENT_REFERENCES"
            )
            if len(tail_text.strip()) > 200 else None
        )
        tf_fut = (
            _content_ex.submit(
                _call,
                TABLES_FIGURES_EXTRACTION_PROMPT,
                full_for_tf,
                tables_figures_max_tokens,
                "CONTENT_TABLES_FIGURES",
            )
            if len(full_for_tf.strip()) > 500 else None
        )
        head_payload = head_fut.result()
        refs_payload = refs_fut.result() if refs_fut is not None else None
        tf_payload = tf_fut.result() if tf_fut is not None else None

    if head_payload:
        for key, bucket in (
            ("body_sections", body_sections),
            ("figure_captions", figure_captions),
            ("table_captions", table_captions),
            ("reference_titles", reference_titles),
        ):
            values = head_payload.get(key) or []
            if isinstance(values, list):
                _dedupe_add(bucket, [str(v).strip() for v in values if str(v).strip()])

    if refs_payload:
        refs = refs_payload.get("references") or []
        if isinstance(refs, list):
            new_titles: List[str] = []
            new_dois: List[str] = []
            for r in refs:
                if not isinstance(r, dict):
                    continue
                title = str(r.get("title") or "").strip()
                if title:
                    new_titles.append(title)
                doi_candidate = str(r.get("doi") or "").strip()
                if doi_candidate:
                    m = _CONTENT_DOI_RE.search(doi_candidate)
                    if m:
                        new_dois.append(m.group(0).lower().rstrip(".,;)"))
            _dedupe_add(reference_titles, new_titles)
            _dedupe_add(reference_dois, new_dois)

    if tf_payload:
        tables_list = tf_payload.get("tables") or []
        figures_list = tf_payload.get("figures") or []
        if isinstance(tables_list, list):
            _dedupe_add(table_captions, [str(t).strip() for t in tables_list if str(t).strip()])
        if isinstance(figures_list, list):
            _dedupe_add(figure_captions, [str(t).strip() for t in figures_list if str(t).strip()])

    return {
        "body_sections": body_sections,
        "figure_captions": figure_captions,
        "table_captions": table_captions,
        "reference_titles": reference_titles,
        "reference_dois": reference_dois,
    }


def merge_content_fields(tei_content: MetadataRecord, llm_content: MetadataRecord) -> MetadataRecord:
    """Merge TEI content_fields with LLM content_fields.

    For each list field the result starts from the TEI list and appends LLM
    items not already present by whitespace-normalised lowercase key.
    Preserves the extractor that found each item first.
    """
    out: MetadataRecord = dict(tei_content)
    for key in ("body_sections", "figure_captions", "table_captions", "reference_titles", "reference_dois"):
        tei_items = list(tei_content.get(key) or [])
        llm_items = list(llm_content.get(key) or [])
        seen = {" ".join(str(i).split()).lower(): None for i in tei_items}
        for item in llm_items:
            key_str = " ".join(str(item).split()).lower()
            if key_str and key_str not in seen:
                tei_items.append(item)
                seen[key_str] = None
        out[key] = tei_items
    return out


def enrich_references_with_crossref(
    pred: MetadataRecord,
    tei_path: Path,
    crossref_client: Any = None,
    max_lookups: int = 80,
    max_workers: int = 5,
) -> MetadataRecord:
    """Augment pred reference fields by looking up any GROBID-parsed biblStruct
    that has a title but no idno DOI via Crossref.

    Mirrors the europe_pmc_jats enrichment pattern in corpus-pipeline: no LLM
    involved; Crossref returns a DOI and a canonical title for matches above
    the client's internal Jaccard threshold. Recovered items are merged into
    pred['reference_dois'] and pred['reference_titles'] with dedup.
    """
    import xml.etree.ElementTree as _ET

    from .crossref import CrossrefClient as _CrossrefClient

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

    preferred_language = detect_language(metadata.get("abstract", ""))
    if preferred_language == "unknown":
        preferred_language = detect_language(metadata.get("title", ""))

    abstract = choose_abstract_candidate(
        [
            metadata.get("abstract", ""),
            header_metadata.get("abstract", ""),
            tei_metadata.get("abstract", ""),
            validated_tei_metadata.get("abstract", ""),
            str(results.get("abstract_from_candidates") or ""),
            ocr_abstract,
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
    if not metadata.get("keywords") and header_metadata.get("keywords"):
        metadata["keywords"] = header_metadata["keywords"]

    multilingual_abstracts = build_multilingual_abstract_blocks(context)
    if multilingual_abstracts:
        metadata["abstract"] = "\n\n".join(multilingual_abstracts)

    targets = keyword_target_languages(metadata.get("keywords") or [], context.lines)
    if targets:
        metadata["keywords"] = translate_keywords(chat, metadata["keywords"], targets)

    metadata["identifiers"] = add_scielo_identifiers(context.record_id, metadata.get("identifiers") or [])
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
        client = AoaiPool(settings.pool_path)
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
