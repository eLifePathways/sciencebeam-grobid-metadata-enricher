# pylint: disable=line-too-long,unused-argument,duplicate-code
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

from grobid_metadata_enricher.pipeline import (
    DocumentContext,
    build_body_section_candidate_evidence,
    build_reference_candidate_evidence_chunks,
    enrich_references_with_crossref,
    figure_caption_candidate_texts,
    merge_content_fields,
    predict_content_fields_from_alto,
    predict_header_metadata,
    prune_layout_lines,
    reference_candidate_texts,
    resolve_field_list,
    resolve_field_text,
    table_caption_candidate_texts,
)


def _line(text: str, page: int = 0, y: float = 100.0, x: float = 72.0) -> Dict[str, Any]:
    return {"text": text, "page": page, "x": x, "y": y}


def _styled_line(
    text: str,
    page: int = 0,
    y: float = 100.0,
    x: float = 72.0,
    *,
    font_size: float = 9.0,
    bold: bool = False,
    h: float = 9.0,
) -> Dict[str, Any]:
    return {"text": text, "page": page, "x": x, "y": y, "font_size": font_size, "bold": bold, "h": h}


def test_prune_layout_lines_removes_repeated_preprint_furniture_but_preserves_reference_doi() -> None:
    lines = []
    for page in range(4):
        lines.extend(
            [
                _line("bioRxiv preprint", page, y=7.0, x=64.0),
                _line("doi:", page, y=7.0, x=122.0),
                _line("https://doi.org/10.1101/2021.02.20.432055", page, y=7.0, x=137.0),
                _line("The copyright holder for this preprint is the author/funder", page, y=15.0, x=68.0),
                _line(str(page + 1), page, y=796.0, x=517.0),
            ]
        )
    lines.extend(
        [
            _line("References", 4, y=80.0),
            _line("Smith A. Useful cited work. doi: 10.1234/example.doi", 4, y=100.0),
        ]
    )

    pruned_text = "\n".join(line["text"] for line in prune_layout_lines(lines))

    assert "bioRxiv preprint" not in pruned_text
    assert "copyright holder" not in pruned_text
    assert "https://doi.org/10.1101/2021.02.20.432055" not in pruned_text
    assert "10.1234/example.doi" in pruned_text


def test_predict_header_metadata_sends_pruned_front_matter_to_llm() -> None:
    lines = []
    for page in range(3):
        lines.extend(
            [
                _line("bioRxiv preprint", page, y=7.0, x=64.0),
                _line("doi:", page, y=7.0, x=122.0),
                _line("https://doi.org/10.1101/example", page, y=7.0, x=137.0),
            ]
        )
    lines.extend(
        [
            _line("A precise article title", 0, y=75.0),
            _line("First Author, Second Author", 0, y=110.0),
        ]
    )
    context = DocumentContext(
        record_id="r1",
        header_text="",
        lines=lines,
        first_page_lines=[line for line in lines if line["page"] == 0],
        tei_fields={},
        tei_abstracts=[],
    )

    def chat(messages: List[Dict[str, str]], **_: Any) -> str:
        user_text = messages[1]["content"]
        assert "bioRxiv preprint" not in user_text
        assert "https://doi.org/10.1101/example" not in user_text
        assert "A precise article title" in user_text
        return json.dumps({"title": "A precise article title", "authors": ["First Author", "Second Author"]})

    assert predict_header_metadata(context, chat)["title"] == "A precise article title"


def test_predict_content_fields_from_alto_sends_pruned_text_to_content_llms() -> None:
    lines = []
    for page in range(4):
        lines.extend(
            [
                _line("bioRxiv preprint", page, y=7.0, x=64.0),
                _line("doi:", page, y=7.0, x=122.0),
                _line("https://doi.org/10.1101/example", page, y=7.0, x=137.0),
                _line(str(page + 1), page, y=796.0, x=517.0),
                _styled_line(
                    "Introduction" if page == 0 else f"Body paragraph block {page}",
                    page,
                    y=85.0,
                    font_size=12.0 if page == 0 else 10.0,
                    bold=page == 0,
                ),
                _styled_line(
                    "This paragraph supplies enough body text for the content extraction passes. " * 8,
                    page,
                    y=120.0,
                    font_size=10.0,
                ),
            ]
        )
    lines.extend(
        [
            _line("Figure 1. A useful caption about the experiment.", 2, y=300.0),
            _line("Table S1. A useful supplemental table caption", 2, y=340.0),
            _line("continued on the next layout line.", 2, y=352.0),
            _line("References", 3, y=500.0),
            _line("Smith A. Useful cited work. doi: 10.1234/example.doi", 3, y=520.0),
        ]
    )
    seen: List[str] = []
    lock = threading.Lock()

    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        user_text = messages[1]["content"]
        with lock:
            seen.append(user_text)
        assert "bioRxiv preprint" not in user_text
        assert "https://doi.org/10.1101/example" not in user_text
        step_name = kwargs["step_name"]
        if step_name == "CONTENT_REFERENCES":
            assert "10.1234/example.doi" in user_text
            return json.dumps({"references": [{"title": "Useful cited work", "doi": "10.1234/example.doi"}]})
        if step_name == "CONTENT_FIGURE_CAPTIONS":
            assert "Figure 1. A useful caption about the experiment." in user_text
            return json.dumps({"figures": ["Figure 1. A useful caption about the experiment."], "tables": []})
        if step_name == "CONTENT_TABLE_CAPTIONS":
            assert "Table S1. A useful supplemental table caption continued on the next layout line." in user_text
            return json.dumps({"tables": ["Table S1. A useful supplemental table caption continued on the next layout line."]})  # noqa: E501
        return json.dumps({"body_sections": ["Introduction"], "figure_captions": [], "table_captions": []})

    result = predict_content_fields_from_alto(
        lines,
        chat,
        max_chars=120,
        references_max_chars=5000,
        tables_figures_max_chars=5000,
    )

    assert result["body_sections"] == ["Introduction"]
    assert result["figure_captions"] == ["Figure 1. A useful caption about the experiment."]
    assert result["table_captions"] == [
        "Table S1. A useful supplemental table caption continued on the next layout line."
    ]
    assert result["reference_dois"] == ["10.1234/example.doi"]
    assert len(seen) == 5


def test_crossref_enrichment_does_not_expand_existing_reference_set() -> None:
    pred = {
        "reference_titles": ["Useful cited work"],
        "reference_dois": ["10.1234/existing"],
    }
    crossref_client = Mock()

    result = enrich_references_with_crossref(pred, Path("missing.tei.xml"), crossref_client=crossref_client)

    assert result is pred
    crossref_client.lookup.assert_not_called()


def test_merge_content_fields_prefers_alto_content_fields_over_tei() -> None:
    result = merge_content_fields(
        {
            "body_sections": ["Introduction", "Noisy TEI heading"],
            "figure_captions": ["Figure 1. TEI caption"],
            "table_captions": ["Table 1. TEI caption"],
            "reference_titles": ["TEI cited work"],
            "reference_dois": ["10.1000/tei"],
        },
        {
            "body_sections": ["Introduction", "Results", "Discussion"],
            "figure_captions": ["Figure 1. ALTO caption"],
            "table_captions": ["Table 1. ALTO caption"],
            "reference_titles": ["ALTO cited work"],
            "reference_dois": ["10.1000/alto"],
        },
    )

    assert result["body_sections"] == ["Introduction", "Results", "Discussion"]
    assert result["figure_captions"] == ["Figure 1. ALTO caption"]
    assert result["table_captions"] == ["Table 1. ALTO caption"]
    # Reference fields union TEI + LLM rather than replace: GROBID's structured
    # extraction often catches DOIs the LLM stage drops at the merge step.
    assert result["reference_titles"] == ["TEI cited work", "ALTO cited work"]
    assert result["reference_dois"] == ["10.1000/tei", "10.1000/alto"]


def test_merge_content_fields_drops_peer_review_dois_keeps_zenodo_with_standard() -> None:
    result = merge_content_fields(
        {
            "reference_dois": [
                "10.21956/openreseurope.19894.r45404",
                "10.5281/zenodo.13785039",
                "10.18174/569408",
            ],
        },
        {
            "reference_dois": [
                "10.5281/zenodo.7300150",
                "10.1016/j.cities.2007.01.009",
            ],
        },
    )

    assert result["reference_dois"] == [
        "10.5281/zenodo.13785039",
        "10.18174/569408",
        "10.5281/zenodo.7300150",
        "10.1016/j.cities.2007.01.009",
    ]


def test_merge_content_fields_drops_zenodo_when_no_standard_dois() -> None:
    result = merge_content_fields(
        {
            "reference_dois": [
                "10.21956/openreseurope.19894.r45404",
                "10.5281/zenodo.13785039",
            ],
        },
        {
            "reference_dois": [
                "10.5281/zenodo.7300150",
            ],
        },
    )

    assert not result["reference_dois"]


def test_body_section_candidates_include_normal_weight_gap_separated_headings_after_intro() -> None:
    lines = [
        _styled_line("Background", page=0, y=90.0, font_size=9.2),
        _styled_line("This is an abstract sentence that should not be selected.", page=0, y=102.0),
        _styled_line("Introduction", page=1, y=90.0, font_size=9.2),
        _styled_line("Body prose starts here and runs as normal paragraph text.", page=1, y=112.0),
        _styled_line("Animals, enrichment types, and handling method", page=1, y=150.0, font_size=9.2),
        _styled_line("The paragraph under that heading starts here.", page=1, y=172.0),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "Introduction" in evidence
    assert "Animals, enrichment types, and handling method" in evidence
    assert "Background" not in evidence


def test_body_section_candidates_merge_adjacent_heading_lines() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=90.0, font_size=11.0, bold=True),
        _styled_line("Single cell RNA sequencing reveals unique transcriptional programs for iPS-derived TS", page=0, y=130.0, font_size=11.0, bold=True),  # noqa: E501
        _styled_line("specification", page=0, y=144.0, font_size=11.0, bold=True),
        _styled_line("The paragraph under that heading starts here.", page=0, y=176.0, font_size=11.0),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "Single cell RNA sequencing reveals unique transcriptional programs for iPS-derived TS specification" in evidence  # noqa: E501
    assert "| specification" not in evidence


def test_body_section_candidates_include_same_font_isolated_heading() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=11.0),
        _styled_line("Body prose begins here and establishes the dominant same-size body font.", page=0, y=105.0, font_size=11.0),  # noqa: E501
        _styled_line("More body prose continues in the same column for spacing context.", page=0, y=117.0, font_size=11.0),  # noqa: E501
        _styled_line("Prediction scoring", page=0, y=168.0, font_size=11.0),
        _styled_line("The paragraph below the isolated heading starts after a visible gap.", page=0, y=194.0, font_size=11.0),  # noqa: E501
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "Prediction scoring" in evidence


def test_body_section_candidates_include_tight_after_gap_methods_heading() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=12.0),
        _styled_line("Normal body prose establishes the dominant body text size.", page=0, y=104.0, font_size=12.0),
        _styled_line("More body prose continues in the same column.", page=0, y=132.0, font_size=12.0),
        _styled_line("Arrestin2 assignment", page=0, y=165.0, font_size=11.0, h=10.0),
        _styled_line("For detecting the interaction, the titration used a narrow same-column gap.", page=0, y=181.0, font_size=12.0),  # noqa: E501
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "Arrestin2 assignment" in evidence


def test_body_section_candidates_reject_same_baseline_table_cells() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, x=72.0, font_size=9.0, bold=True),
        _styled_line("Table 1. Questionnaire summary.", page=0, y=120.0, x=72.0, font_size=8.0, bold=True),
        _styled_line("Questionnaire", page=0, y=150.0, x=92.0, font_size=8.0, bold=True),
        _styled_line("Type", page=0, y=150.0, x=320.0, font_size=8.0, bold=True),
        _styled_line("Total number of items", page=0, y=150.0, x=380.0, font_size=8.0, bold=True),
        _styled_line("Results", page=0, y=230.0, x=72.0, font_size=9.0, bold=True),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "Results" in evidence
    assert "Questionnaire" not in evidence
    assert "Total number of items" not in evidence


def test_table_caption_candidates_stop_before_table_header_rows() -> None:
    lines = [
        _styled_line("Table 1. siRNAs used in this study.", page=0, y=80.0, x=72.0, font_size=12.0, bold=True),
        _styled_line("siRNA name", page=0, y=95.0, x=72.0, font_size=10.0, bold=True),
        _styled_line("Source", page=0, y=95.0, x=220.0, font_size=10.0, bold=True),
        _styled_line("Catalog", page=0, y=95.0, x=320.0, font_size=10.0, bold=True),
        _styled_line("si scramble control", page=0, y=108.0, x=72.0, font_size=10.0),
    ]

    assert table_caption_candidate_texts(lines) == ["Table 1. siRNAs used in this study."]


def test_table_caption_candidates_stop_before_following_section_heading() -> None:
    lines = [
        _styled_line("Table 3: Tensorflow-/Numpy-like pseudocode of peak", page=0, y=80.0, x=72.0, font_size=10.0),
        _styled_line("matching and scoring for PSMs.", page=0, y=92.0, x=72.0, font_size=10.0),
        _styled_line("yHydra error-tolerant search via gradient descent", page=0, y=114.0, x=72.0, font_size=12.0),
    ]

    assert table_caption_candidate_texts(lines) == [
        "Table 3: Tensorflow-/Numpy-like pseudocode of peak matching and scoring for PSMs."
    ]


def test_figure_caption_candidates_join_continuation_but_stop_before_table_caption() -> None:
    lines = [
        _styled_line("Figure 1. Main assay overview and representative images.", page=0, y=80.0, x=72.0),
        _styled_line("Error bars indicate standard deviation across replicates.", page=0, y=92.0, x=72.0),
        _styled_line("Table 1. Reagents used in this study.", page=0, y=120.0, x=72.0),
    ]

    assert figure_caption_candidate_texts(lines) == [
        "Figure 1. Main assay overview and representative images. Error bars indicate standard deviation across replicates."  # noqa: E501
    ]


def test_reference_candidates_are_bounded_to_reference_section() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0),
        _styled_line("This line cites Smith but is not a reference entry.", page=0, y=100.0),
        _styled_line("References", page=3, y=80.0),
        _styled_line("1. Smith A. Useful cited work. Journal 2020. doi: 10.1234/example", page=3, y=100.0),
        _styled_line("2. Jones B. Another cited work. Journal 2021.", page=3, y=120.0),
    ]

    assert reference_candidate_texts(lines) == [
        "1. Smith A. Useful cited work. Journal 2020. doi: 10.1234/example",
        "2. Jones B. Another cited work. Journal 2021.",
    ]


def test_reference_candidates_read_two_column_references_column_by_column() -> None:
    lines = [
        _styled_line("References", page=0, y=80.0, x=72.0),
        _styled_line("1. Left A. First left title.", page=0, y=100.0, x=72.0),
        _styled_line("3. Right A. First right title.", page=0, y=100.0, x=320.0),
        _styled_line("Journal left continuation.", page=0, y=112.0, x=72.0),
        _styled_line("Journal right continuation.", page=0, y=112.0, x=320.0),
        _styled_line("2. Left B. Second left title.", page=0, y=126.0, x=72.0),
        _styled_line("4. Right B. Second right title.", page=0, y=126.0, x=320.0),
        _styled_line("Journal left two.", page=0, y=138.0, x=72.0),
        _styled_line("Journal right two.", page=0, y=138.0, x=320.0),
    ]

    assert reference_candidate_texts(lines) == [
        "1. Left A. First left title. Journal left continuation.",
        "2. Left B. Second left title. Journal left two.",
        "3. Right A. First right title. Journal right continuation.",
        "4. Right B. Second right title. Journal right two.",
    ]


def test_reference_candidate_evidence_chunks_split_long_bibliography() -> None:
    lines = [_styled_line("References", page=0, y=80.0)]
    for i in range(120):
        lines.append(_styled_line(
            f"{i + 1}. Author{i}, B. Title of work {i} on relevant topic. Journal Name {2000 + i}.",
            page=0,
            y=100.0 + i * 12.0,
        ))

    chunks = build_reference_candidate_evidence_chunks(lines, chunk_size=50)

    assert len(chunks) == 3
    assert chunks[0].splitlines()[0].startswith("[1] ")
    assert chunks[0].splitlines()[-1].startswith("[50] ")
    assert chunks[1].splitlines()[0].startswith("[51] ")
    assert chunks[1].splitlines()[-1].startswith("[100] ")
    assert chunks[2].splitlines()[0].startswith("[101] ")
    assert chunks[2].splitlines()[-1].startswith("[120] ")


def test_reference_candidate_evidence_chunks_returns_empty_when_no_references() -> None:
    lines = [_styled_line("Introduction", page=0, y=80.0)]

    assert not build_reference_candidate_evidence_chunks(lines, chunk_size=50)


def test_reference_section_heading_survives_repeated_furniture_pruning() -> None:
    # Long bibliographies repeat 'References' as a running header on continuation
    # pages; pruning treated the heading as repeated furniture, leaving the
    # reference detector with nothing to anchor to.
    lines = [
        _styled_line("Body paragraph one with substantive sentence content here.", page=0, y=400.0),
        _styled_line("References", page=1, y=80.0),
        _styled_line(
            "Smith A, Jones B, Davis C: Useful cited work on the topic. Journal Name. 2020.",
            page=1,
            y=120.0,
        ),
        _styled_line("References", page=2, y=80.0),
        _styled_line(
            "Smith A, Jones B, Davis C: Useful cited work on the topic. Journal Name. 2020.",
            page=2,
            y=120.0,
        ),
        _styled_line("References", page=3, y=80.0),
        _styled_line(
            "Smith A, Jones B, Davis C: Useful cited work on the topic. Journal Name. 2020.",
            page=3,
            y=120.0,
        ),
    ]

    pruned = prune_layout_lines(lines)
    assert any(line["text"] == "References" for line in pruned)


def test_reference_start_index_picks_earliest_when_heading_repeats() -> None:
    from grobid_metadata_enricher.pipeline import _reference_start_index
    lines = [
        _styled_line("References", page=0, y=80.0),
        _styled_line(
            "Smith A: First reference of the article bibliography. Journal. 2020.",
            page=0,
            y=120.0,
        ),
        _styled_line("References", page=2, y=80.0),  # reviewer report bibliography
    ]

    assert _reference_start_index(lines) == 0


def test_predict_content_fields_preserves_distinct_similar_table_captions() -> None:
    lines = [
        _styled_line(
            "Table 3. Selection of questionnaire subdimensions recommended to achieve a more solid epistemic comparability.",  # noqa: E501
            page=0,
            y=80.0,
        ),
        _styled_line(
            "Table 4. Selection of questionnaire items recommended to achieve a more solid epistemic comparability.",
            page=0,
            y=110.0,
        ),
    ]

    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if kwargs["step_name"] == "CONTENT_TABLE_CAPTIONS":
            return json.dumps(
                {
                    "tables": [
                        "Table 3. Selection of questionnaire subdimensions recommended to achieve a more solid epistemic comparability.",  # noqa: E501
                        "Table 4. Selection of questionnaire items recommended to achieve a more solid epistemic comparability.",  # noqa: E501
                    ]
                }
            )
        return json.dumps({"body_sections": [], "figures": [], "tables": [], "references": []})

    result = predict_content_fields_from_alto(lines, chat, max_chars=10, references_max_chars=10, tables_figures_max_chars=10)  # noqa: E501

    assert result["table_captions"] == [
        "Table 3. Selection of questionnaire subdimensions recommended to achieve a more solid epistemic comparability.",  # noqa: E501
        "Table 4. Selection of questionnaire items recommended to achieve a more solid epistemic comparability.",
    ]


def test_predict_content_fields_rejects_unsupported_figure_and_reference_llm_outputs() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=12.0, bold=True),
    ]
    for i in range(80):
        lines.append(
            _styled_line(
                f"This is normal article body prose line {i} with enough text to trigger content extraction.",
                page=0,
                y=100.0 + i * 12,
            )
        )

    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if kwargs["step_name"] == "CONTENT_FIGURE_CAPTIONS":
            return json.dumps({"figures": ["Figure 9. Invented unsupported caption."]})
        if kwargs["step_name"] == "CONTENT_REFERENCES":
            return json.dumps(
                {"references": [{"title": "Invented Unsupported Reference Title", "doi": "10.9999/invented"}]}
            )
        return json.dumps({"body_sections": [], "tables": []})

    result = predict_content_fields_from_alto(
        lines,
        chat,
        max_chars=10,
        references_max_chars=5000,
        tables_figures_max_chars=5000,
    )

    assert not result["figure_captions"]
    assert not result["reference_titles"]
    assert not result["reference_dois"]


def test_body_section_candidates_do_not_merge_from_prose_into_heading() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=9.2),
        _styled_line("Critical appraisal of individual sources of evidence", page=0, y=130.0, font_size=9.2),
        _styled_line("From a preliminary screening, I found that some items inquire", page=0, y=142.0, font_size=9.0),
        _styled_line("about unrelated details in the prose paragraph.", page=0, y=154.0, font_size=9.0),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "Critical appraisal of individual sources of evidence" in evidence
    assert "Critical appraisal of individual sources of evidence From" not in evidence


def test_body_section_candidates_trim_inline_heading_sentence() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=10.0, bold=True),
        _styled_line(
            "Bipartite parameterization of DEIs. Thus far, we have described the model in qualitative terms.",
            page=0,
            y=130.0,
            font_size=10.0,
            bold=True,
        ),
        _styled_line("The paragraph under that heading starts here.", page=0, y=160.0, font_size=10.0),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "Bipartite parameterization of DEIs" in evidence
    assert "Thus far" not in evidence


def test_body_section_candidates_stop_after_back_matter_heading() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=10.0, bold=True),
        _styled_line("Discussion", page=0, y=120.0, font_size=10.0, bold=True),
        _styled_line("Acknowledgements", page=0, y=160.0, font_size=10.0, bold=True),
        _styled_line("Materials and Methods", page=0, y=200.0, font_size=10.0, bold=True),
        _styled_line("Cell culture", page=0, y=230.0, font_size=10.0, bold=True),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "Discussion" in evidence
    assert "Acknowledgements" not in evidence
    assert "Materials and Methods" not in evidence
    assert "Cell culture" not in evidence


def test_body_section_candidates_keep_major_heading_separate_from_subheading() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=11.0, bold=True),
        _styled_line("MATERIALS AND METHODS", page=0, y=130.0, font_size=11.0, bold=True),
        _styled_line("HeLa, C2C12, and betaTC6 cell culture", page=0, y=144.0, font_size=11.0, bold=True),
        _styled_line("The paragraph under that heading starts here.", page=0, y=176.0, font_size=11.0),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "MATERIALS AND METHODS" in evidence
    assert "HeLa, C2C12, and betaTC6 cell culture" in evidence
    assert "MATERIALS AND METHODS HeLa" not in evidence


def test_body_section_candidates_merge_connector_continuation_lines() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=10.0, bold=True),
        _styled_line("yHydra boosts peptide identifications for", page=0, y=130.0, font_size=10.0, bold=True),
        _styled_line("a monoclonal antibody via error-tolerant", page=0, y=144.0, font_size=10.0, bold=True),
        _styled_line("searching", page=0, y=158.0, font_size=10.0, bold=True),
        _styled_line("The paragraph under that heading starts here.", page=0, y=190.0, font_size=10.0),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "yHydra boosts peptide identifications for a monoclonal antibody via error-tolerant searching" in evidence
    assert "| a monoclonal antibody" not in evidence


def test_body_section_candidates_merge_small_bold_hyphen_continuation() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=10.0, bold=True),
        _styled_line("Normal body prose establishes the dominant body font size.", page=0, y=110.0, font_size=10.0),
        _styled_line("The MICrONS dataset reveals synaptic connectivity re-", page=0, y=150.0, font_size=9.5, bold=True),  # noqa: E501
        _styled_line("flecting a functional invariance hierarchy in V1 Layer", page=0, y=162.0, font_size=9.5, bold=True),  # noqa: E501
        _styled_line("2/3. To gain insights into the prose paragraph", page=0, y=174.0, font_size=10.0, bold=True),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "The MICrONS dataset reveals synaptic connectivity reflecting a functional invariance hierarchy in V1 Layer 2/3" in evidence  # noqa: E501
    assert "To gain insights" not in evidence


def test_body_section_candidates_include_small_bold_headings_near_body_font() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=9.5, bold=True),
        _styled_line("Normal body prose establishes the dominant body text size.", page=0, y=104.0, font_size=9.5),
        _styled_line("Additional normal body prose for the body-font mode.", page=0, y=116.0, font_size=9.5),
        _styled_line("Training a DeepLabCut network model", page=0, y=150.0, font_size=8.5, bold=True),
        _styled_line("The paragraph below that heading starts here.", page=0, y=174.0, font_size=9.5),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "Training a DeepLabCut network model" in evidence


def test_predict_content_fields_preserves_numeric_body_section_prefixes() -> None:
    lines = [
        _styled_line("1. Introduction", page=0, y=90.0, font_size=11.0, bold=True),
        _styled_line("2. Methods", page=0, y=130.0, font_size=11.0, bold=True),
        _styled_line("2.1 Plant material", page=0, y=160.0, font_size=11.0, bold=True),
        _styled_line("2.2 Field experiments 2.2.1 Field experiment 2022", page=0, y=190.0, font_size=11.0, bold=True),
        _styled_line("3. Results", page=0, y=220.0, font_size=11.0, bold=True),
    ]

    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if kwargs["step_name"] == "CONTENT_BODY_SECTIONS":
            return json.dumps(
                {
                    "body_sections": [
                        "1. Introduction",
                        "2. Methods",
                        "2.1 Plant material",
                        "2.2 Field experiments 2.2.1 Field experiment 2022",
                        "3. Results",
                    ]
                }
            )
        return json.dumps({"body_sections": []})

    result = predict_content_fields_from_alto(lines, chat, max_chars=1000)

    assert result["body_sections"] == [
        "1. Introduction",
        "2. Methods",
        "2.1 Plant material",
        "2.2 Field experiments",
        "2.2.1 Field experiment 2022",
        "3. Results",
    ]


def test_body_section_candidates_use_document_line_spacing_to_reject_paragraph_starts() -> None:
    lines = [
        _styled_line("INTRODUCTION", page=0, y=74.0, x=108.0, font_size=12.0),
        _styled_line(
            "Mammals, including humans, fail to regenerate neurons lost to protracted",
            page=0,
            y=110.0,
            x=108.0,
            font_size=12.0,
        ),
        _styled_line("secondary cell death around a spinal injury site.", page=0, y=138.0, x=72.0, font_size=12.0),
        _styled_line("RESULTS", page=1, y=74.0, x=108.0, font_size=12.0),
        _styled_line(
            "sema4ab-dependent microglia signalling indicated by scRNA-seq",
            page=1,
            y=110.0,
            x=108.0,
            font_size=12.0,
            bold=True,
        ),
        _styled_line("The paragraph below that subheading starts here.", page=1, y=138.0, x=72.0, font_size=12.0),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "candidate: INTRODUCTION" in evidence
    assert "candidate: RESULTS" in evidence
    assert "candidate: sema4ab-dependent microglia signalling indicated by scRNA-seq" in evidence
    assert "candidate: Mammals, including humans" not in evidence
    assert "following_text:" in evidence


def test_body_section_candidates_reject_numbered_procedure_items() -> None:
    lines = [
        _styled_line("1. Introduction", page=0, y=80.0, font_size=10.0, bold=True),
        _styled_line("Normal body prose establishes the dominant body font size.", page=0, y=110.0, font_size=10.0),
        _styled_line("1. Planula collection: collect larvae from traps", page=0, y=150.0, font_size=10.0, bold=True),
        _styled_line("2. Methods", page=0, y=190.0, font_size=10.0, bold=True),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "candidate: 1. Introduction" in evidence
    assert "candidate: 2. Methods" in evidence
    assert "Planula collection" not in evidence


def test_body_section_candidates_reject_small_numbered_footnotes() -> None:
    lines = [
        _styled_line("1. Introduction", page=0, y=80.0, font_size=10.0, bold=True),
        _styled_line("Normal body prose establishes the dominant body font size.", page=0, y=110.0, font_size=10.0),
        _styled_line("More body prose establishes the body font size.", page=0, y=122.0, font_size=10.0),
        _styled_line("1. This is a small-font footnote line", page=0, y=150.0, font_size=7.0, bold=True),
        _styled_line("VII", page=0, y=190.0, font_size=12.0, bold=True),
        _styled_line("The paragraph under the roman heading starts here.", page=0, y=230.0, font_size=10.0),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "candidate: 1. Introduction" in evidence
    assert "candidate: VII" in evidence
    assert "candidate: 1. This is a small-font footnote line" not in evidence


def test_body_section_candidates_reject_body_font_bold_prose_without_section_gap() -> None:
    lines = [
        _styled_line("1. Introduction", page=0, y=80.0, font_size=10.0, bold=True),
        _styled_line("Normal body prose establishes the dominant body font size.", page=0, y=110.0, font_size=10.0),
        _styled_line("More body prose continues in the same column.", page=0, y=122.0, font_size=10.0),
        _styled_line("Via this route the scheme can provide a complementary incentive", page=0, y=134.0, font_size=10.0, bold=True),  # noqa: E501
        _styled_line("and the paragraph continues immediately below.", page=0, y=146.0, font_size=10.0),
        _styled_line("2. Methods", page=0, y=190.0, font_size=10.0, bold=True),
        _styled_line("The paragraph under Methods starts here.", page=0, y=222.0, font_size=10.0),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "candidate: 1. Introduction" in evidence
    assert "candidate: 2. Methods" in evidence
    assert "candidate: Via this route the scheme can provide a complementary incentive" not in evidence


def test_body_section_candidates_preserve_colon_and_short_sentence_headings() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=10.0, bold=True),
        _styled_line("Normal body prose establishes the dominant body font size.", page=0, y=110.0, font_size=10.0),
        _styled_line(
            "Discussion: Lessons for policymaking integration",
            page=0,
            y=150.0,
            font_size=10.0,
            bold=True,
        ),
        _styled_line("and policy coherence", page=0, y=162.0, font_size=10.0, bold=True),
        _styled_line("The paragraph under that heading starts here.", page=0, y=194.0, font_size=10.0),
        _styled_line(
            "Take these cautionary tales seriously. Don’t wish them",
            page=0,
            y=240.0,
            font_size=10.0,
            bold=True,
        ),
        _styled_line("away then reinvent the wheel", page=0, y=252.0, font_size=10.0, bold=True),
        _styled_line("The next paragraph starts here.", page=0, y=284.0, font_size=10.0),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "candidate: Discussion: Lessons for policymaking integration and policy coherence" in evidence
    assert "candidate: Take these cautionary tales seriously. Don’t wish them away then reinvent the wheel" in evidence


def test_body_section_candidates_merge_question_heading_continuation() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=11.0, bold=True),
        _styled_line("Normal body prose establishes the dominant body font size.", page=0, y=112.0, font_size=9.0),
        _styled_line(
            "The need for conceptual clarity: what exactly does",
            page=0,
            y=152.0,
            font_size=9.2,
        ),
        _styled_line(
            "policymaking integration mean?",
            page=0,
            y=163.0,
            font_size=9.2,
        ),
        _styled_line("The paragraph under that heading starts here.", page=0, y=196.0, font_size=9.0),
    ]

    evidence = build_body_section_candidate_evidence(lines)

    assert "candidate: The need for conceptual clarity: what exactly does policymaking integration mean?" in evidence


def test_body_section_candidates_relaxed_layout_recovers_biorxiv_small_headings() -> None:
    lines = [
        _styled_line("Introduction", page=0, y=80.0, font_size=12.0, bold=True),
        _styled_line("Normal body prose establishes the dominant body font size.", page=0, y=112.0, font_size=12.0),
        _styled_line("More normal body prose continues in the same column.", page=0, y=124.0, font_size=12.0),
        _styled_line("Methods", page=0, y=190.0, font_size=18.0),
        _styled_line("Experimental Design", page=0, y=222.0, font_size=10.1),
        _styled_line("The paragraph under that heading starts here.", page=0, y=254.0, font_size=12.0),
    ]

    strict_evidence = build_body_section_candidate_evidence(lines, use_document_spacing=False)
    relaxed_evidence = build_body_section_candidate_evidence(
        lines,
        use_document_spacing=False,
        relaxed_layout=True,
    )

    assert "candidate: Experimental Design" not in strict_evidence
    assert "candidate: Experimental Design" in relaxed_evidence


def test_predict_content_fields_runs_legacy_number_stripping_pass_for_biorxiv_layout() -> None:
    lines = [
        _styled_line("bioRxiv preprint", page=0, y=8.0, x=72.0, font_size=8.0),
        _styled_line("1 Introduction", page=0, y=80.0, x=72.0, font_size=12.0, bold=True),
        _styled_line("Introductory body text starts here.", page=0, y=112.0, x=72.0, font_size=12.0),
        _styled_line("2 Methods", page=0, y=170.0, x=72.0, font_size=12.0, bold=True),
        _styled_line("Methods body text starts here.", page=0, y=202.0, x=72.0, font_size=12.0),
    ]
    body_prompts: List[str] = []

    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if kwargs["step_name"] == "CONTENT_BODY_SECTIONS":
            body_prompts.append(messages[1]["content"])
            return json.dumps({"body_sections": ["1 Introduction", "2 Methods"]})
        return json.dumps({"body_sections": []})

    result = predict_content_fields_from_alto(lines, chat, max_chars=1000)

    assert result["body_sections"] == ["Introduction", "Methods"]
    assert len(body_prompts) == 2
    assert any("following_text:" in prompt for prompt in body_prompts)
    assert any("following_text:" not in prompt for prompt in body_prompts)


def test_resolve_field_text_joins_and_dehyphenates_valid_indices() -> None:
    lines = [
        {"text": "Ethics in the publication of"},
        {"text": "studies on human visceral"},
        {"text": "leishmaniasis in Brazilian"},
        {"text": "periodicals"},
    ]
    result = resolve_field_text("LLM-claimed text", [1, 2, 3, 4], lines)
    assert result == "Ethics in the publication of studies on human visceral leishmaniasis in Brazilian periodicals"


def test_resolve_field_text_dehyphenates_line_break_words() -> None:
    lines = [{"text": "experi-"}, {"text": "ment was successful"}]
    assert resolve_field_text("fallback", [1, 2], lines) == "experiment was successful"


def test_resolve_field_text_falls_back_on_invalid_indices() -> None:
    lines = [{"text": "alpha"}, {"text": "beta"}]
    assert resolve_field_text("FALLBACK", [], lines) == "FALLBACK"
    assert resolve_field_text("FALLBACK", [1, 99], lines) == "FALLBACK"
    assert resolve_field_text("FALLBACK", ["1", "2"], lines) == "FALLBACK"
    assert resolve_field_text("FALLBACK", [1, 1, 2], lines) == "FALLBACK"


def test_resolve_field_list_reconstructs_per_item_groups() -> None:
    lines = [{"text": "Author One"}, {"text": "Author Two"}, {"text": "Author Three"}]
    assert resolve_field_list(["A", "B", "C"], [[1], [2], [3]], lines) == [
        "Author One", "Author Two", "Author Three"
    ]


def test_resolve_field_list_joins_multi_line_groups_with_dehyphenation() -> None:
    lines = [{"text": "Maria"}, {"text": "Andrea Yáñez"}, {"text": "John Smith"}]
    assert resolve_field_list(
        ["Maria Andrea Yáñez", "John Smith"], [[1, 2], [3]], lines
    ) == ["Maria Andrea Yáñez", "John Smith"]


def test_resolve_field_list_falls_back_per_item_on_invalid_group() -> None:
    lines = [{"text": "alpha"}, {"text": "beta"}]
    # Empty groups -> use parsed_items wholesale
    assert resolve_field_list(["A", "B"], [], lines) == ["A", "B"]
    # Per-item fallback when one group is invalid
    assert resolve_field_list(["KEEP_ME", "FALLBACK"], [[1], [99]], lines) == ["alpha", "FALLBACK"]
    # Non-list group falls back per item
    assert resolve_field_list(["KEEP_ME", "FALLBACK"], [[1], "garbage"], lines) == ["alpha", "FALLBACK"]
