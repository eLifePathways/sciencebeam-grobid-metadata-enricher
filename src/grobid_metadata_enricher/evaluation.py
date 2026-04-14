from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

WORD_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)


def normalize_text(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[^a-z0-9' ]+", "", value)


def normalize_tokens(text: str) -> List[str]:
    return [token.lower() for token in WORD_RE.findall(text or "")]


def author_tokens(name: str) -> List[str]:
    stopwords = {"de", "da", "do", "dos", "das", "e", "y", "del", "la", "los"}
    return [token for token in normalize_tokens(name) if token not in stopwords]


def author_match(gold: str, predicted_authors: List[str]) -> bool:
    gold_tokens = author_tokens(gold)
    if not gold_tokens:
        return False
    gold_last_name = gold_tokens[-1]
    for predicted in predicted_authors:
        predicted_tokens = set(author_tokens(predicted))
        if predicted_tokens and gold_last_name in predicted_tokens and predicted_tokens.intersection(gold_tokens):
            return True
    return False


def jaccard_recall(gold: str, predicted: str) -> float:
    gold_tokens = set(normalize_tokens(gold))
    predicted_tokens = set(normalize_tokens(predicted))
    if not gold_tokens:
        return 1.0
    return len(gold_tokens & predicted_tokens) / max(1, len(gold_tokens))


def keyword_recall(gold: List[str], predicted: List[str]) -> float:
    gold_values = {normalize_text(value) for value in gold if normalize_text(value)}
    predicted_values = {normalize_text(value) for value in predicted if normalize_text(value)}
    if not gold_values:
        return 1.0
    return len(gold_values & predicted_values) / max(1, len(gold_values))


def scalar_match(gold: str, predicted: str) -> Optional[int]:
    gold_value = normalize_text(gold)
    if not gold_value:
        return None
    return 1 if gold_value == normalize_text(predicted) else 0


def normalize_identifier(value: str) -> str:
    return value.strip().lower().replace("doi:", "").strip()


def identifier_recall(gold: List[str], predicted: List[str]) -> float:
    gold_values = {normalize_identifier(value) for value in gold if value.strip()}
    predicted_values = {normalize_identifier(value) for value in predicted if value.strip()}
    if not gold_values:
        return 1.0
    return len(gold_values & predicted_values) / max(1, len(gold_values))


def language_match(gold: str, predicted: str) -> Optional[int]:
    if not gold:
        return None
    aliases = {"pt": "por", "en": "eng", "es": "spa"}
    gold_value = aliases.get(normalize_text(gold), normalize_text(gold))
    predicted_value = aliases.get(normalize_text(predicted), normalize_text(predicted))
    if not predicted_value:
        return 0
    return 1 if gold_value == predicted_value else 0


def evaluate_record(predicted: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    predicted_title = normalize_text(predicted.get("title", ""))
    gold_titles = gold.get("titles") or [gold.get("title", "")]
    metrics["title_match"] = 1 if any(predicted_title == normalize_text(title) for title in gold_titles if title) else 0

    predicted_authors = predicted.get("authors") or []
    gold_authors = gold.get("authors") or []
    if gold_authors:
        matches = sum(1 for author in gold_authors if author_match(author, predicted_authors))
        metrics["authors_recall"] = matches / max(1, len(gold_authors))
    else:
        metrics["authors_recall"] = 1.0

    predicted_abstract = predicted.get("abstract", "")
    gold_abstracts = gold.get("abstracts") or [gold.get("abstract", "")]
    metrics["abstract_recall"] = (
        max(jaccard_recall(abstract, predicted_abstract) for abstract in gold_abstracts)
        if gold_abstracts
        else jaccard_recall(gold.get("abstract", ""), predicted_abstract)
    )

    predicted_keywords = predicted.get("keywords") or []
    gold_keyword_groups = gold.get("keywords_groups") or {}
    if gold_keyword_groups:
        recalls = [keyword_recall(group, predicted_keywords) for group in gold_keyword_groups.values() if group]
        fallback = keyword_recall(gold.get("keywords") or [], predicted_keywords)
        metrics["keywords_recall"] = max(recalls) if recalls else fallback
    else:
        metrics["keywords_recall"] = keyword_recall(gold.get("keywords") or [], predicted_keywords)

    metrics["publisher_match"] = scalar_match(gold.get("publisher", ""), predicted.get("publisher", ""))
    metrics["date_match"] = scalar_match(gold.get("date", ""), predicted.get("date", ""))
    metrics["language_match"] = language_match(gold.get("language", ""), predicted.get("language", ""))
    metrics["rights_match"] = scalar_match(gold.get("rights", ""), predicted.get("rights", ""))
    metrics["types_recall"] = keyword_recall(gold.get("types") or [], predicted.get("types") or [])
    metrics["formats_recall"] = keyword_recall(gold.get("formats") or [], predicted.get("formats") or [])
    metrics["relations_recall"] = keyword_recall(gold.get("relations") or [], predicted.get("relations") or [])
    metrics["identifiers_recall"] = identifier_recall(gold.get("identifiers") or [], predicted.get("identifiers") or [])
    return metrics


def aggregate_metrics(per_document: List[Dict[str, Any]]) -> Dict[str, Any]:
    values: Dict[str, List[float]] = {}
    for row in per_document:
        for metric, value in row["metrics"].items():
            if value is None:
                continue
            values.setdefault(metric, []).append(float(value))
    summary: Dict[str, Any] = {"n": len(per_document)}
    for metric, metric_values in values.items():
        summary[metric] = sum(metric_values) / max(1, len(metric_values))
    return summary


def shorten(text: str, max_length: int = 160) -> str:
    value = str(text or "").strip().replace("\n", " ")
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def write_root_cause_report(
    per_document: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [f"# {title}", ""]
    lines.append(f"Context: SciELO preprints (n={summary.get('n', 0)}) extracted vs OAI-DC gold.")
    lines.append("")
    metric_order = [
        "title_match",
        "authors_recall",
        "abstract_recall",
        "keywords_recall",
        "publisher_match",
        "date_match",
        "language_match",
        "identifiers_recall",
        "relations_recall",
        "rights_match",
        "types_recall",
        "formats_recall",
    ]
    present_metrics = [f"{metric}~{summary[metric]:.3f}" for metric in metric_order if metric in summary]
    if present_metrics:
        lines.append("Key metrics:")
        lines.append(", ".join(present_metrics))
        lines.append("")

    def examples(predicate: Callable[[Dict[str, Any]], bool], limit: int = 2) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        for row in per_document:
            if predicate(row):
                matches.append(row)
            if len(matches) >= limit:
                break
        return matches

    lines.append("## Title")
    missing_title = examples(lambda row: normalize_text(row["pred"].get("title", "")) == "")
    long_title = examples(
        lambda row: len(row["pred"].get("title", "")) > max(1, len(row["gold"].get("title", "")) * 1.5)
    )
    short_title = examples(
        lambda row: len(row["pred"].get("title", "")) < max(10, len(row["gold"].get("title", "")) * 0.6)
    )
    if missing_title:
        lines.append("Root cause: missing or empty title output.")
        for row in missing_title:
            lines.append(
                f"- {row['record_id']}: gold='{shorten(row['gold'].get('title', ''))}'"
                f" pred='{shorten(row['pred'].get('title', ''))}'"
            )
    if long_title:
        lines.append("Root cause: title includes extra lines or disclaimer text.")
        for row in long_title:
            lines.append(f"- {row['record_id']}: pred='{shorten(row['pred'].get('title', ''))}'")
    if short_title:
        lines.append("Root cause: title truncated or incomplete.")
        for row in short_title:
            lines.append(
                f"- {row['record_id']}: gold='{shorten(row['gold'].get('title', ''))}'"
                f" pred='{shorten(row['pred'].get('title', ''))}'"
            )
    lines.append("")

    lines.append("## Authors")
    missing_authors = examples(lambda row: not (row["pred"].get("authors") or []))
    noisy_authors = examples(
        lambda row: any(
            token in author.lower()
            for author in row["pred"].get("authors") or []
            for token in ("orcid", "http", "univers", "institute")
        )
    )
    if missing_authors:
        lines.append("Root cause: authors missing from extraction.")
        for row in missing_authors:
            lines.append(f"- {row['record_id']}: gold={row['gold'].get('authors', [])}")
    if noisy_authors:
        lines.append("Root cause: author list polluted with affiliation noise.")
        for row in noisy_authors:
            lines.append(f"- {row['record_id']}: pred={row['pred'].get('authors', [])}")
    lines.append("")

    lines.append("## Abstract")
    missing_abstract = examples(lambda row: normalize_text(row["pred"].get("abstract", "")) == "")
    short_abstract = examples(
        lambda row: len(row["pred"].get("abstract", "")) < max(20, len(row["gold"].get("abstract", "")) * 0.3)
    )
    if missing_abstract:
        lines.append("Root cause: abstract missing.")
        for row in missing_abstract:
            lines.append(f"- {row['record_id']}: gold='{shorten(row['gold'].get('abstract', ''))}'")
    if short_abstract:
        lines.append("Root cause: abstract truncated.")
        for row in short_abstract:
            lines.append(
                f"- {row['record_id']}: gold='{shorten(row['gold'].get('abstract', ''))}'"
                f" pred='{shorten(row['pred'].get('abstract', ''))}'"
            )
    lines.append("")

    lines.append("## Keywords")
    missing_keywords = examples(lambda row: not (row["pred"].get("keywords") or []))
    partial_keywords = examples(lambda row: row["metrics"].get("keywords_recall", 1.0) < 0.5)
    if missing_keywords:
        lines.append("Root cause: keywords missing from extraction.")
        for row in missing_keywords:
            lines.append(f"- {row['record_id']}: gold={row['gold'].get('keywords', [])}")
    if partial_keywords:
        lines.append("Root cause: keywords partially recovered.")
        for row in partial_keywords:
            lines.append(
                f"- {row['record_id']}: gold={row['gold'].get('keywords', [])} pred={row['pred'].get('keywords', [])}"
            )
    lines.append("")

    lines.append("## Publisher / Date / Rights / Types / Formats / Relations")
    metadata_gaps = examples(
        lambda row: not row["pred"].get("publisher") or not row["pred"].get("date") or not row["pred"].get("rights")
    )
    if metadata_gaps:
        lines.append("Root cause: metadata fields remain empty.")
        for row in metadata_gaps:
            lines.append(
                f"- {row['record_id']}: publisher='{shorten(row['pred'].get('publisher', ''))}'"
                f" date='{shorten(row['pred'].get('date', ''))}'"
                f" rights='{shorten(row['pred'].get('rights', ''))}'"
            )
    lines.append("")

    lines.append("## Language")
    language_mismatches = examples(lambda row: row["metrics"].get("language_match") == 0)
    if language_mismatches:
        lines.append("Root cause: language code mismatch.")
        for row in language_mismatches:
            lines.append(
                f"- {row['record_id']}: gold='{row['gold'].get('language', '')}'"
                f" pred='{row['pred'].get('language', '')}'"
            )
    lines.append("")

    lines.append("## Identifiers")
    missing_identifiers = examples(lambda row: not (row["pred"].get("identifiers") or []))
    if missing_identifiers:
        lines.append("Root cause: identifiers missing or incomplete.")
        for row in missing_identifiers:
            lines.append(
                f"- {row['record_id']}: gold={row['gold'].get('identifiers', [])}"
                f" pred={row['pred'].get('identifiers', [])}"
            )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
