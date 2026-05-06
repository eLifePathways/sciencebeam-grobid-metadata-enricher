from __future__ import annotations

import json
from typing import Any, Dict, List

from grobid_metadata_enricher.pipeline import (
    DocumentContext,
    _build_prediction_inner,
    _keyword_marker_evidence_text,
    choose_abstract_candidate,
    choose_abstract_candidate_from_sources,
    extract_abstract_from_ocr,
    marker_windows,
    normalize_identifier_values,
    normalize_keyword_values,
)


def _line(text: str, page: int = 0) -> dict:
    return {"text": text, "page": page, "x": 0.0, "y": 0.0}


def _context(**tei_fields: Any) -> DocumentContext:
    lines = [
        _line("Title"),
        _line("Resumo"),
        _line("Resumo em portugues."),
        _line("Abstract"),
        _line("This is the selected English abstract."),
        _line("Keywords: covid-19; mortality"),
    ]
    return DocumentContext(
        record_id="oai_ops.preprints.scielo.org_preprint_2459",
        header_text="",
        lines=lines,
        first_page_lines=lines,
        tei_fields={
            "title": "Effects of COVID-19 on mortality",
            "authors": [],
            "abstract": "",
            "keywords": [],
            "identifiers": [],
            "language": "en",
            **tei_fields,
        },
        tei_abstracts=[
            "Resumo: Resumo em portugues.",
            "Abstract: This is the selected English abstract.",
        ],
    )


def _chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
    step_name = kwargs.get("step_name")
    if step_name == "HEADER_METADATA":
        return json.dumps(
            {
                "title": "Effects of COVID-19 on mortality",
                "authors": [],
                "affiliations": [],
                "abstract": "",
                "keywords": ["hospital mortality", "covid-19"],
            }
        )
    if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
        return json.dumps(
            {
                "title": "Effects of COVID-19 on mortality",
                "authors": [],
                "affiliations": [],
                "abstract": "",
                "keywords": ["mortalidade hospitalar", "covid-19"],
                "identifiers": [
                    "10.1016/j.cell.2014.07.051",
                    "10.1590/SciELOPreprints.2459",
                    "0000-0001-2345-6789",
                ],
            }
        )
    if step_name == "ABSTRACT_SELECT":
        return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
    if step_name == "ABSTRACT_FROM_OCR":
        assert "This is the selected English abstract" in messages[1]["content"]
        return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
    if step_name == "KEYWORD_EXTRACT":
        assert "Keywords: covid-19; mortality" in messages[1]["content"]
        return json.dumps(
            {
                "keyword_lists": [
                    {"language": "en", "keywords": ["covid-19", "mortality"]},
                ]
            }
        )
    if step_name == "KEYWORD_SELECT":
        assert "hospital mortality" in messages[1]["content"]
        return json.dumps({"keywords": ["mortalidade hospitalar", "covid-19"]})
    if step_name == "IDENTIFIER_SELECT":
        assert "10.1016/j.cell.2014.07.051" in messages[1]["content"]
        return json.dumps(
            {
                "identifiers": [
                    "https://preprints.scielo.org/index.php/scielo/preprint/view/2459",
                    "10.1590/SciELOPreprints.2459",
                ]
            }
        )
    if step_name == "KEYWORD_TRANSLATE":
        raise AssertionError("keyword translation union should not run")
    raise AssertionError(f"unexpected step: {step_name}")


def test_llm_abstract_selection_is_not_overwritten_by_multilingual_union() -> None:
    result = _build_prediction_inner(_context(), _chat, per_document_llm_workers=1)

    assert result["abstract"] == "Abstract: This is the selected English abstract."
    assert "\n\n" not in result["abstract"]
    assert "Resumo em portugues" not in result["abstract"]


def test_abstract_candidate_selection_rejects_conflict_and_funding_disclosures() -> None:
    scientific = (
        "Resumo: Objetivo: avaliar a possível semelhança entre sequências de aminoácidos. "
        "Método: estudo de imunoinformática com comparação de epítopos. "
        "Resultados: foram identificadas regiões candidatas. "
        "Conclusão: os achados orientam novas análises experimentais."
    )

    assert choose_abstract_candidate(
        [
            "Divulgação de potenciais conflitos de interesse: Nenhum dos autores tem conflitos a divulgar.",
            "Fuentes de financiamiento: Autofinanciado. Conflictos de interés: no hay.",
            scientific,
        ],
        "pt",
    ) == scientific


def test_supported_pdf_abstract_selection_respects_metadata_language() -> None:
    abstract = (
        "Resumo: Este resumo em portugues deve ser mantido quando a lingua do artigo e pt. "
        "O estudo descreve objetivos, metodos, resultados e conclusoes com detalhes suficientes "
        "para ser identificado como o resumo principal do artigo."
    )
    context = _context(abstract=abstract, language="pt")
    lines = [
        _line("Titulo"),
        _line("Resumo"),
        _line(abstract.removeprefix("Resumo: ")),
        _line("Palavras-chave: mortalidade; covid-19"),
    ]
    context = DocumentContext(
        record_id=context.record_id,
        header_text=context.header_text,
        lines=lines,
        first_page_lines=lines,
        tei_fields=context.tei_fields,
        tei_abstracts=[abstract],
    )
    result = _build_prediction_inner(
        context,
        _chat,
        per_document_llm_workers=1,
    )

    assert result["abstract"] == abstract


def test_consensus_abstract_selection_rejects_unsupported_tei_body_bleed() -> None:
    wrong_tei = (
        "Introduction This long body section discusses laboratory setup, materials, reagents, "
        "centrifugation, sample handling, and procedural details that GROBID mislabeled as an abstract. "
        "It continues with implementation details and therefore should not win only because it is long."
    )
    correct = (
        "Protein-RNA interactions underpin many critical biological processes, demanding the development "
        "of technologies to precisely characterize their nature and functions. Here, we present an improved "
        "workflow that increases detected cross-link products and improves structural modelling."
    )

    assert choose_abstract_candidate_from_sources(
        [
            ("tei_fields", wrong_tei),
            ("abstract_from_candidates", correct[:160]),
            ("ocr_abstract", correct),
            ("header_metadata", correct),
        ],
        "en",
    ) == correct


def test_llm_keyword_selection_replaces_translation_union() -> None:
    result = _build_prediction_inner(
        _context(keywords=["mortalidade hospitalar", "covid-19"]),
        _chat,
        per_document_llm_workers=1,
    )

    assert result["keywords"] == ["mortalidade hospitalar", "covid-19"]
    assert "hospital mortality" not in result["keywords"]


def test_llm_keyword_extraction_recovers_front_matter_keywords_when_structured_sources_miss() -> None:
    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name == "ABSTRACT_SELECT":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps(
                {"keyword_lists": [{"language": "en", "keywords": ["covid-19", "mortality"]}]}
            )
        if step_name == "IDENTIFIER_SELECT":
            return json.dumps({"identifiers": ["https://preprints.scielo.org/index.php/scielo/preprint/view/2459"]})
        raise AssertionError(f"unexpected step: {step_name}")

    result = _build_prediction_inner(_context(), chat, per_document_llm_workers=1)

    assert result["keywords"] == ["covid-19", "mortality"]


def test_llm_keyword_extraction_can_supply_clean_atomic_keywords_for_selector() -> None:
    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name == "ABSTRACT_SELECT":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps(
                {
                    "keyword_lists": [
                        {
                            "language": "en",
                            "keywords": [
                                "Tobacco use disorder",
                                "Education",
                                "Education, nursing, continuing",
                                "Education, distance",
                            ],
                        }
                    ]
                }
            )
        if step_name == "KEYWORD_SELECT":
            return json.dumps(
                {
                    "keywords": [
                        "Tobacco use disorder",
                        "Education",
                        "Education, nursing, continuing",
                        "Education, distance",
                    ]
                }
            )
        if step_name == "IDENTIFIER_SELECT":
            return json.dumps({"identifiers": ["https://preprints.scielo.org/index.php/scielo/preprint/view/2459"]})
        raise AssertionError(f"unexpected step: {step_name}")

    result = _build_prediction_inner(
        _context(keywords=["Tobacco use disorder. Education", "nursing", "continuing. Education", "distance"]),
        chat,
        per_document_llm_workers=1,
    )

    assert result["keywords"] == [
        "Tobacco use disorder",
        "Education",
        "Education, nursing, continuing",
        "Education, distance",
    ]


def test_single_tei_keyword_list_is_validated_and_can_be_rejected() -> None:
    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name == "ABSTRACT_SELECT":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps({"keyword_lists": []})
        if step_name == "KEYWORD_SELECT":
            assert "Conflict of interest" in messages[1]["content"]
            return json.dumps({"keywords": []})
        if step_name == "IDENTIFIER_SELECT":
            return json.dumps({"identifiers": ["https://preprints.scielo.org/index.php/scielo/preprint/view/2459"]})
        raise AssertionError(f"unexpected step: {step_name}")

    result = _build_prediction_inner(
        _context(keywords=["Conflict of interest", "no Funding", "none"]),
        chat,
        per_document_llm_workers=1,
    )

    assert result["keywords"] == []


def test_compact_tei_keyword_list_without_marker_is_preserved_when_not_suspect() -> None:
    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name == "ABSTRACT_SELECT":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps({"keyword_lists": []})
        if step_name == "KEYWORD_SELECT":
            return json.dumps({"keywords": ["COVID-19", "mortality"]})
        if step_name == "IDENTIFIER_SELECT":
            return json.dumps({"identifiers": ["https://preprints.scielo.org/index.php/scielo/preprint/view/2459"]})
        raise AssertionError(f"unexpected step: {step_name}")

    base = _context(keywords=["COVID-19", "mortality"])
    no_keyword_marker_context = DocumentContext(
        record_id=base.record_id,
        header_text=base.header_text,
        lines=[_line("Title"), _line("Abstract"), _line("This is the selected English abstract.")],
        first_page_lines=[_line("Title"), _line("Abstract"), _line("This is the selected English abstract.")],
        tei_fields=base.tei_fields,
        tei_abstracts=base.tei_abstracts,
    )

    result = _build_prediction_inner(no_keyword_marker_context, chat, per_document_llm_workers=1)

    assert result["keywords"] == ["COVID-19", "mortality"]


def test_keyword_inference_runs_only_when_no_keyword_candidates_or_markers_exist() -> None:
    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps({"title": "Plate fixation versus intramedullary fixation", "authors": [], "affiliations": [], "abstract": "", "keywords": []})  # noqa: E501
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps({"title": "Plate fixation versus intramedullary fixation", "authors": [], "affiliations": [], "abstract": "", "keywords": []})  # noqa: E501
        if step_name == "ABSTRACT_SELECT":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps({"keyword_lists": []})
        if step_name == "KEYWORD_SELECT":
            return json.dumps({"keywords": []})
        if step_name == "KEYWORD_INFER":
            assert "Plate fixation versus intramedullary fixation" in messages[1]["content"]
            return json.dumps({"keywords": ["Bone fractures", "Clavicle", "Plate fixation", "Intramedullary fixation"]})
        if step_name == "IDENTIFIER_SELECT":
            return json.dumps({"identifiers": ["https://preprints.scielo.org/index.php/scielo/preprint/view/2459"]})
        raise AssertionError(f"unexpected step: {step_name}")

    base = _context()
    context = DocumentContext(
        record_id=base.record_id,
        header_text=base.header_text,
        lines=[_line("Plate fixation versus intramedullary fixation"), _line("Abstract"), _line("This is the selected English abstract.")],  # noqa: E501
        first_page_lines=[_line("Plate fixation versus intramedullary fixation"), _line("Abstract"), _line("This is the selected English abstract.")],  # noqa: E501
        tei_fields={**base.tei_fields, "keywords": [], "title": "Plate fixation versus intramedullary fixation"},
        tei_abstracts=base.tei_abstracts,
    )

    result = _build_prediction_inner(context, chat, per_document_llm_workers=1)

    assert result["keywords"] == ["Bone fractures", "Clavicle", "Plate fixation", "Intramedullary fixation"]


def test_front_matter_keywords_win_when_selector_underselects_non_front_matter_subset() -> None:
    front_keywords = [
        "quarantine",
        "positive interventions",
        "subjective well-being",
        "creativity",
        "positive psychology",
    ]

    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps(
                {"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": ["COVID-19"]}
            )
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps(
                {
                    "title": "Title",
                    "authors": [],
                    "affiliations": [],
                    "abstract": "",
                    "keywords": ["COVID-19", "Positive Psychology"],
                }
            )
        if step_name == "ABSTRACT_SELECT":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps({"keyword_lists": [{"language": "en", "keywords": front_keywords}]})
        if step_name == "KEYWORD_SELECT":
            return json.dumps({"keywords": ["COVID-19", "Positive Psychology"]})
        if step_name == "IDENTIFIER_SELECT":
            return json.dumps({"identifiers": ["https://preprints.scielo.org/index.php/scielo/preprint/view/2459"]})
        raise AssertionError(f"unexpected step: {step_name}")

    result = _build_prediction_inner(_context(keywords=["COVID-19", "Positive Psychology"]), chat, 1)

    assert result["keywords"] == front_keywords


def test_keyword_normalization_strips_and_splits_embedded_labels() -> None:
    assert normalize_keyword_values(
        [
            "Palabras clave Genetic diversity",
            "public databases",
            "Immunity Palavras Chave: Vacinas",
            "Pessoal de Saúde",
        ]
    ) == [
        "Genetic diversity",
        "public databases",
        "Immunity",
        "Vacinas",
        "Pessoal de Saúde",
    ]


def test_keyword_normalization_splits_polluted_tei_keyword_blob_before_author_roles() -> None:
    assert normalize_keyword_values(
        [
            (
                "living evidence, evidence synthesis, living evidence framework, health decisions, "
                "decision making, knowledge transfer, capacity building, heath systems research "
                "Author roles: Rojas-Reyes MX: Conceptualization, Funding Acquisition"
            ),
            "Urrutia Chuchi G: Methodology, Supervision, Writing -Review & Editing",
        ]
    ) == [
        "living evidence",
        "evidence synthesis",
        "living evidence framework",
        "health decisions",
        "decision making",
        "knowledge transfer",
        "capacity building",
        "heath systems research",
    ]


def test_marker_windows_keep_structured_abstract_headings_and_skip_prefix_boilerplate() -> None:
    lines = [
        _line("bioRxiv preprint doi: https://doi.org/10.1101/example", page=2),
        _line("Abstract", page=2),
        _line("Background", page=2),
        _line("New regulations on low emission vehicles incentivized lightweight composites.", page=2),
        _line("Methods", page=2),
        _line("We evaluated solvolysis under mild conditions.", page=2),
        _line("Keywords", page=2),
        _line("SMC; Solvolysis", page=2),
    ]

    assert marker_windows(lines, max_blocks=1, prefix_lines=0, suffix_lines=20, fallback_lines=20) == [
        (
            "Abstract: Background New regulations on low emission vehicles incentivized lightweight composites. "
            "Methods We evaluated solvolysis under mild conditions."
        )
    ]


def test_keyword_marker_evidence_searches_early_front_matter_pages_not_only_page_zero() -> None:
    lines = [
        _line("Title", page=0),
        _line("Author list", page=0),
        _line("Abstract", page=1),
        _line("This is the article abstract.", page=1),
        _line("Keywords", page=1),
        _line("FAIR, FAIR evaluators, FAIRness assessment, Governance", page=1),
    ]
    context = DocumentContext(
        record_id="2-146_v2",
        header_text="",
        lines=lines,
        first_page_lines=[line for line in lines if line["page"] == 0],
        tei_fields={},
        tei_abstracts=[],
    )

    evidence = _keyword_marker_evidence_text(context)

    assert "Keywords" in evidence
    assert "FAIR evaluators" in evidence


def test_bilingual_abstract_block_can_be_segmented_before_selection() -> None:
    spanish = (
        "El test SCL-90-R es utilizado para evaluar diversos sintomas psicopatologicos. "
        "En la presente investigacion, dicha tecnica fue administrada a distancia mediante un formulario online. "
        "La recoleccion de datos se da en contexto de aislamiento social preventivo y obligatorio por COVID-19. "
        "Las dimensiones principales mostraron elevados niveles de consistencia interna."
    )
    english = (
        "The SCL-90-R test is used to evaluate and describe various psychopathological symptoms. "
        "In the present investigation, this technique was administered remotely using an online form. "
        "The compilation of current data occurs in the context of social preventive and compulsory isolation by COVID-19. "  # noqa: E501
        "The nine main dimensions showed high levels of internal consistency."
    )

    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name == "ABSTRACT_SELECT":
            assert "source=tei_abstract_segment" in messages[1]["content"]
            assert english in messages[1]["content"]
            return json.dumps({"abstract": f"{spanish} {english}"})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": english})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps({"keyword_lists": []})
        if step_name == "IDENTIFIER_SELECT":
            return json.dumps({"identifiers": ["https://preprints.scielo.org/index.php/scielo/preprint/view/2459"]})
        raise AssertionError(f"unexpected step: {step_name}")

    context = _context(abstract=f"{spanish} {english}", language="en")
    context = DocumentContext(
        record_id=context.record_id,
        header_text=context.header_text,
        lines=context.lines,
        first_page_lines=context.first_page_lines,
        tei_fields=context.tei_fields,
        tei_abstracts=[f"{spanish} {english}"],
    )

    result = _build_prediction_inner(context, chat, per_document_llm_workers=1)

    assert result["abstract"] == english
    assert spanish not in result["abstract"]


def test_abstract_from_ocr_rejects_summary_not_supported_by_source_text() -> None:
    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        assert kwargs.get("step_name") == "ABSTRACT_FROM_OCR"
        return json.dumps(
            {
                "abstract": (
                    "This study summarizes policy decisions and reports broad implications that are not "
                    "extractively present in the supplied text."
                )
            }
        )

    assert extract_abstract_from_ocr("Status: Preprint submitted. Title only. Authors only.", chat) == ""


def test_llm_identifier_selection_drops_reference_ids_and_orcids() -> None:
    result = _build_prediction_inner(
        _context(
            identifiers=[
                "10.1016/j.cell.2014.07.051",
                "10.1590/SciELOPreprints.2459",
                "0000-0001-2345-6789",
            ]
        ),
        _chat,
        per_document_llm_workers=1,
    )

    assert result["identifiers"] == [
        "https://preprints.scielo.org/index.php/scielo/preprint/view/2459",
        "10.1590/SciELOPreprints.2459",
    ]


def test_llm_identifier_selection_can_prefer_official_article_doi_over_generated_preprint_doi() -> None:
    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps(
                {
                    "title": "Title",
                    "authors": [],
                    "affiliations": [],
                    "abstract": "",
                    "keywords": [],
                    "identifiers": ["10.1590/2236-8906-81/2020"],
                }
            )
        if step_name == "ABSTRACT_SELECT":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps({"keyword_lists": []})
        if step_name == "IDENTIFIER_SELECT":
            assert "10.1590/SciELOPreprints.2459" in messages[1]["content"]
            return json.dumps(
                {
                    "identifiers": [
                        "https://preprints.scielo.org/index.php/scielo/preprint/view/2459",
                        "10.1590/2236-8906-81/2020",
                    ]
                }
            )
        raise AssertionError(f"unexpected step: {step_name}")

    result = _build_prediction_inner(
        _context(identifiers=["10.1590/2236-8906-81/2020"]),
        chat,
        per_document_llm_workers=1,
    )

    assert result["identifiers"] == [
        "https://preprints.scielo.org/index.php/scielo/preprint/view/2459",
        "10.1590/2236-8906-81/2020",
    ]


def test_identifier_selection_keeps_evidenced_preprint_doi_with_published_article_doi() -> None:
    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name == "ABSTRACT_SELECT":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps({"keyword_lists": []})
        if step_name == "IDENTIFIER_SELECT":
            return json.dumps({"identifiers": ["10.61679/1503015026"]})
        raise AssertionError(f"unexpected step: {step_name}")

    base = _context(identifiers=["10.61679/1503015026"])
    lines = [
        _line("DOI del artículo publicado: https://doi.org/10.61679/1503015026"),
        _line("https://doi.org/10.1590/SciELOPreprints.354"),
        *_context().lines,
    ]
    context = DocumentContext(
        record_id="oai_ops.preprints.scielo.org_preprint_354",
        header_text=base.header_text,
        lines=lines,
        first_page_lines=lines,
        tei_fields=base.tei_fields,
        tei_abstracts=base.tei_abstracts,
    )

    result = _build_prediction_inner(context, chat, per_document_llm_workers=1)

    assert result["identifiers"] == [
        "https://preprints.scielo.org/index.php/scielo/preprint/view/354",
        "10.61679/1503015026",
        "10.1590/SciELOPreprints.354",
    ]


def test_identifier_selection_does_not_invent_preprint_doi_when_front_matter_says_not_informed() -> None:
    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name == "ABSTRACT_SELECT":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps({"keyword_lists": []})
        if step_name == "IDENTIFIER_SELECT":
            return json.dumps({"identifiers": ["10.36660/ijcs.20200057"]})
        raise AssertionError(f"unexpected step: {step_name}")

    base = _context(identifiers=["10.36660/ijcs.20200057"])
    lines = [
        _line("DOI of the published article: https://doi.org/10.36660/ijcs.20200057"),
        _line("https://doi.org/Not informed"),
        *_context().lines,
    ]
    context = DocumentContext(
        record_id="oai_ops.preprints.scielo.org_preprint_63",
        header_text=base.header_text,
        lines=lines,
        first_page_lines=lines,
        tei_fields=base.tei_fields,
        tei_abstracts=base.tei_abstracts,
    )

    result = _build_prediction_inner(context, chat, per_document_llm_workers=1)

    assert result["identifiers"] == [
        "https://preprints.scielo.org/index.php/scielo/preprint/view/63",
        "10.36660/ijcs.20200057",
    ]


def test_llm_identifier_selection_adds_scielo_landing_url_when_model_returns_only_doi() -> None:
    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name == "ABSTRACT_SELECT":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps({"keyword_lists": []})
        if step_name == "IDENTIFIER_SELECT":
            return json.dumps({"identifiers": ["10.1590/SciELOPreprints.2459"]})
        raise AssertionError(f"unexpected step: {step_name}")

    result = _build_prediction_inner(_context(), chat, per_document_llm_workers=1)

    assert result["identifiers"] == [
        "https://preprints.scielo.org/index.php/scielo/preprint/view/2459",
        "10.1590/SciELOPreprints.2459",
    ]


def test_identifier_candidates_normalize_malformed_doi_prefixes_before_selection() -> None:
    assert normalize_identifier_values(["em:10.1590/S1679-49742021000200006."]) == [
        "10.1590/S1679-49742021000200006"
    ]

    def chat(messages: List[Dict[str, str]], **kwargs: Any) -> str:
        step_name = kwargs.get("step_name")
        if step_name == "HEADER_METADATA":
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name in {"TEI_METADATA", "TEI_VALIDATED"}:
            return json.dumps({"title": "Title", "authors": [], "affiliations": [], "abstract": "", "keywords": []})
        if step_name == "ABSTRACT_SELECT":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "ABSTRACT_FROM_OCR":
            return json.dumps({"abstract": "Abstract: This is the selected English abstract."})
        if step_name == "KEYWORD_EXTRACT":
            return json.dumps({"keyword_lists": []})
        if step_name == "IDENTIFIER_SELECT":
            assert "em:10.1590" not in messages[1]["content"]
            assert "10.1590/S1679-49742021000200006" in messages[1]["content"]
            return json.dumps({"identifiers": ["10.1590/S1679-49742021000200006"]})
        raise AssertionError(f"unexpected step: {step_name}")

    base = _context(identifiers=["em:10.1590/S1679-49742021000200006"])
    context = DocumentContext(
        record_id="oai_ops.preprints.scielo.org_preprint_1637",
        header_text=base.header_text,
        lines=base.lines,
        first_page_lines=base.first_page_lines,
        tei_fields=base.tei_fields,
        tei_abstracts=base.tei_abstracts,
    )

    result = _build_prediction_inner(context, chat, per_document_llm_workers=1)

    assert result["identifiers"] == [
        "https://preprints.scielo.org/index.php/scielo/preprint/view/1637",
        "10.1590/S1679-49742021000200006",
    ]
