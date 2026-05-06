from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rapidfuzz.distance import Levenshtein as _Levenshtein

PRTriple = Tuple[Optional[float], Optional[float], Optional[float]]

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
    normalized = value.strip().lower().replace("doi:", "").strip()
    if normalized.startswith("10."):
        normalized = re.sub(r"\s+", "", normalized)
    return normalized


def identifier_recall(gold: List[str], predicted: List[str]) -> float:
    gold_values = {normalize_identifier(value) for value in gold if value.strip()}
    predicted_values = {normalize_identifier(value) for value in predicted if value.strip()}
    if not gold_values:
        return 1.0
    return len(gold_values & predicted_values) / max(1, len(gold_values))


def levenshtein_sim(a: str, b: str) -> float:
    return _Levenshtein.normalized_similarity(a or "", b or "")


def normalized_edit_sim(a: str, b: str) -> float:
    return levenshtein_sim(normalize_text(a), normalize_text(b))


def language_match(gold: str, predicted: str) -> Optional[int]:
    if not gold:
        return None
    aliases = {"pt": "por", "en": "eng", "es": "spa"}
    gold_value = aliases.get(normalize_text(gold), normalize_text(gold))
    predicted_value = aliases.get(normalize_text(predicted), normalize_text(predicted))
    if not predicted_value:
        return 0
    return 1 if gold_value == predicted_value else 0


def _set_pr(gold: set, pred: set) -> PRTriple:
    if not gold:
        return (None, None, None)
    inter = len(gold & pred)
    rec = inter / len(gold)
    pre = inter / len(pred) if pred else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
    return (pre, rec, f1)


def _bipartite_pr(
    n_gold: int,
    n_pred: int,
    is_match: Callable[[int, int], bool],
) -> PRTriple:
    if n_gold == 0:
        return (None, None, None)
    pred_used = [False] * n_pred
    matched = 0
    for gi in range(n_gold):
        for pi in range(n_pred):
            if pred_used[pi]:
                continue
            if is_match(gi, pi):
                pred_used[pi] = True
                matched += 1
                break
    rec = matched / n_gold
    pre = matched / n_pred if n_pred else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
    return (pre, rec, f1)


def _abstract_pr(gold: str, predicted: str) -> PRTriple:
    sim = normalized_edit_sim(gold, predicted)
    return (sim, sim, sim)


def _keyword_pr(gold: List[str], predicted: List[str]) -> PRTriple:
    g = {normalize_text(v) for v in (gold or []) if normalize_text(v)}
    p = {normalize_text(v) for v in (predicted or []) if normalize_text(v)}
    return _set_pr(g, p)


def _identifier_pr(gold: List[str], predicted: List[str]) -> PRTriple:
    g = {normalize_identifier(v) for v in (gold or []) if v and v.strip()}
    p = {normalize_identifier(v) for v in (predicted or []) if v and v.strip()}
    g.discard("")
    p.discard("")
    return _set_pr(g, p)


def _edit_similarity_match(gold: str, pred: str, threshold: float) -> bool:
    if not normalize_text(gold) or not normalize_text(pred):
        return False
    return normalized_edit_sim(gold, pred) >= threshold


def _section_head_pr(gold_heads: List[str], pred_heads: List[str], threshold: float = 0.7) -> PRTriple:
    golds = [h for h in (gold_heads or []) if normalize_text(h)]
    preds = [h for h in (pred_heads or []) if normalize_text(h)]
    return _bipartite_pr(
        len(golds),
        len(preds),
        lambda gi, pi: _edit_similarity_match(golds[gi], preds[pi], threshold),
    )


def _caption_set_pr(gold_captions: List[str], pred_captions: List[str]) -> PRTriple:
    golds = [c for c in (gold_captions or []) if normalize_text(c)]
    preds = [c for c in (pred_captions or []) if normalize_text(c)]
    if not golds:
        return (None, None, None)
    return _bipartite_pr(
        len(golds),
        len(preds),
        lambda gi, pi: normalized_edit_sim(golds[gi], preds[pi]) >= 0.75,
    )


def _reference_pr(gold: Dict[str, Any], predicted: Dict[str, Any]) -> PRTriple:
    gold_dois = {normalize_identifier(d) for d in (gold.get("reference_dois") or []) if d}
    pred_dois = {normalize_identifier(d) for d in (predicted.get("reference_dois") or []) if d}
    gold_dois.discard("")
    pred_dois.discard("")
    if gold_dois and pred_dois:
        return _set_pr(gold_dois, pred_dois)
    gold_titles = [t for t in (gold.get("reference_titles") or []) if t]
    if not gold_titles:
        return (None, None, None)
    pred_titles = [t for t in (predicted.get("reference_titles") or []) if t]
    return _section_head_pr(gold_titles, pred_titles, threshold=0.5)


def _reference_combined_pr(
    gold: Dict[str, Any],
    predicted: Dict[str, Any],
    edit_threshold: float = 0.5,
) -> PRTriple:
    records = gold.get("reference_records") or []
    if not records:
        return (None, None, None)
    pred_doi_set = {normalize_identifier(str(d)) for d in (predicted.get("reference_dois") or []) if d}
    pred_doi_set.discard("")
    pred_titles = [str(t) for t in (predicted.get("reference_titles") or []) if t]

    used_doi: set = set()
    used_title = [False] * len(pred_titles)
    matched = 0
    for r in records:
        doi = normalize_identifier(str(r.get("doi") or ""))
        title = str(r.get("title") or "")
        if doi and doi in pred_doi_set and doi not in used_doi:
            used_doi.add(doi)
            matched += 1
            continue
        if not title:
            continue
        title_norm = normalize_text(title)
        if not title_norm:
            continue
        best_i = -1
        best_sim = -1.0
        for i, pt in enumerate(pred_titles):
            if used_title[i]:
                continue
            sim = normalized_edit_sim(title_norm, pt)
            if sim > best_sim:
                best_sim = sim
                best_i = i
        if best_i >= 0 and best_sim >= edit_threshold:
            used_title[best_i] = True
            matched += 1

    pred_total = len(pred_doi_set) + len(pred_titles)
    rec = matched / len(records)
    pre = matched / pred_total if pred_total else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
    return (pre, rec, f1)


def _assign_pr(metrics: Dict[str, Any], name: str, triple: PRTriple) -> None:
    pre, rec, f1 = triple
    metrics[f"{name}_precision"] = pre
    metrics[f"{name}_recall"] = rec
    metrics[f"{name}_f1"] = f1


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
    gold_abstracts = [a for a in (gold.get("abstracts") or [gold.get("abstract", "")]) if a]
    if gold_abstracts:
        prs = [_abstract_pr(g, predicted_abstract) for g in gold_abstracts]
        edits = [levenshtein_sim(g, predicted_abstract) for g in gold_abstracts]
        best = max(prs, key=lambda t: t[2] or -1.0)
        metrics["abstract_precision"] = best[0]
        metrics["abstract_recall"] = best[1]
        metrics["abstract_f1"] = best[2]
        metrics["abstract_edit_sim"] = max(edits)
    else:
        metrics["abstract_precision"] = None
        metrics["abstract_recall"] = None
        metrics["abstract_f1"] = None
        metrics["abstract_edit_sim"] = None

    predicted_keywords = predicted.get("keywords") or []
    gold_keyword_groups = gold.get("keywords_groups") or {}
    if gold_keyword_groups:
        prs = [_keyword_pr(group, predicted_keywords) for group in gold_keyword_groups.values() if group]
        prs = [t for t in prs if t[1] is not None]
        if prs:
            best = max(prs, key=lambda t: t[2] or -1.0)
            _assign_pr(metrics, "keywords", best)
        else:
            _assign_pr(metrics, "keywords", _keyword_pr(gold.get("keywords") or [], predicted_keywords))
    else:
        _assign_pr(metrics, "keywords", _keyword_pr(gold.get("keywords") or [], predicted_keywords))

    metrics["publisher_match"] = scalar_match(gold.get("publisher", ""), predicted.get("publisher", ""))
    metrics["date_match"] = scalar_match(gold.get("date", ""), predicted.get("date", ""))
    metrics["language_match"] = language_match(gold.get("language", ""), predicted.get("language", ""))
    metrics["rights_match"] = scalar_match(gold.get("rights", ""), predicted.get("rights", ""))
    _assign_pr(metrics, "types", _keyword_pr(gold.get("types") or [], predicted.get("types") or []))
    _assign_pr(metrics, "formats", _keyword_pr(gold.get("formats") or [], predicted.get("formats") or []))
    _assign_pr(metrics, "relations", _keyword_pr(gold.get("relations") or [], predicted.get("relations") or []))
    _assign_pr(metrics, "identifiers",
               _identifier_pr(gold.get("identifiers") or [], predicted.get("identifiers") or []))

    if "body_sections" in gold:
        _assign_pr(metrics, "body_section",
                   _section_head_pr(gold.get("body_sections") or [], predicted.get("body_sections") or []))
    if "figure_captions" in gold:
        _assign_pr(metrics, "figure_caption",
                   _caption_set_pr(gold.get("figure_captions") or [], predicted.get("figure_captions") or []))
    if "table_captions" in gold:
        _assign_pr(metrics, "table_caption",
                   _caption_set_pr(gold.get("table_captions") or [], predicted.get("table_captions") or []))
    if "reference_dois" in gold or "reference_titles" in gold:
        _assign_pr(metrics, "reference", _reference_pr(gold, predicted))
    if "reference_records" in gold:
        _assign_pr(metrics, "reference_combined", _reference_combined_pr(gold, predicted))
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
        "abstract_f1",
        "abstract_edit_sim",
        "keywords_f1",
        "publisher_match",
        "date_match",
        "language_match",
        "identifiers_f1",
        "relations_f1",
        "rights_match",
        "types_f1",
        "formats_f1",
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
