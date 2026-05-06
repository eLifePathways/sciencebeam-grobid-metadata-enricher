from __future__ import annotations

from pathlib import Path

from grobid_metadata_enricher.formats import extract_tei_abstracts, extract_tei_fields


def test_extract_tei_abstracts_splits_structured_header_abstract_and_stops_before_body(tmp_path: Path) -> None:
    tei_path = tmp_path / "doc.tei.xml"
    tei_path.write_text(
        """
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title>Example</title></titleStmt>
              <profileDesc>
                <abstract>
                  <div><p>Objective: Identify the real abstract. Method: Structured review.</p></div>
                  <div><head>Conclusion:</head><p>The result is complete. Keywords: One. Two.</p></div>
                  <div><head>RESUMEN</head><p>Objetivo: Otro resumen. Palabras clave: Tres. Cuatro.</p></div>
                  <div><head>INTRODUCTION</head><p>This must not bleed into the abstract.</p></div>
                </abstract>
              </profileDesc>
            </fileDesc>
          </teiHeader>
        </TEI>
        """,
        encoding="utf-8",
    )

    assert extract_tei_abstracts(tei_path) == [
        "Objective: Identify the real abstract. Method: Structured review. The result is complete.",
        "Objetivo: Otro resumen.",
    ]
    fields = extract_tei_fields(tei_path)
    assert fields["abstract"] == (
        "Objective: Identify the real abstract. Method: Structured review. The result is complete."
    )
    assert fields["keywords"] == ["One", "Two", "Tres", "Cuatro"]


def test_extract_tei_fields_uses_body_abstract_when_header_is_disclosure(tmp_path: Path) -> None:
    tei_path = tmp_path / "doc.tei.xml"
    tei_path.write_text(
        """
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title>Example</title></titleStmt>
              <profileDesc>
                <abstract><p>Conflicts of interest: authors declare no conflict.</p></abstract>
              </profileDesc>
            </fileDesc>
          </teiHeader>
          <text>
            <body>
              <div><head>ABSTRACT</head><p>This is the actual article abstract.</p>
              <p>Key-words: multimorbidity, Covid-19, behavior</p></div>
            </body>
          </text>
        </TEI>
        """,
        encoding="utf-8",
    )

    fields = extract_tei_fields(tei_path)

    assert fields["abstract"] == "This is the actual article abstract."
    assert fields["keywords"] == ["multimorbidity", "Covid-19", "behavior"]


def test_extract_tei_abstracts_keeps_inline_abstract_head_text(tmp_path: Path) -> None:
    tei_path = tmp_path / "doc.tei.xml"
    tei_path.write_text(
        """
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <div>
                <head>RESUMO Objective: Keep the text embedded in the heading</head>
                <p>and append the paragraph text without dropping the start.</p>
              </div>
              <div><head>Introdução</head><p>This must not be included.</p></div>
            </body>
          </text>
        </TEI>
        """,
        encoding="utf-8",
    )

    assert extract_tei_abstracts(tei_path) == [
        "Objective: Keep the text embedded in the heading and append the paragraph text without dropping the start."
    ]


def test_extract_tei_abstracts_uses_body_lead_paragraph_when_no_abstract_label(tmp_path: Path) -> None:
    tei_path = tmp_path / "doc.tei.xml"
    tei_path.write_text(
        """
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <div>
                <head>Key findings</head>
                <p>This report links transport-flow data and demographic data with confirmed disease cases
                to provide strategic information about geographic propagation. It identifies regions that
                may operate as spread centers and regions that are vulnerable to receiving infected people.
                The summary is intentionally long enough to look like an article-level synopsis rather than
                a short heading or a table-of-contents line.</p>
              </div>
              <div><head>Methods</head><p>This later section must not be used.</p></div>
            </body>
          </text>
        </TEI>
        """,
        encoding="utf-8",
    )

    assert extract_tei_abstracts(tei_path) == [
        "This report links transport-flow data and demographic data with confirmed disease cases "
        "to provide strategic information about geographic propagation. It identifies regions that "
        "may operate as spread centers and regions that are vulnerable to receiving infected people. "
        "The summary is intentionally long enough to look like an article-level synopsis rather than "
        "a short heading or a table-of-contents line."
    ]


def test_extract_tei_abstracts_groups_structured_body_abstract_sections(tmp_path: Path) -> None:
    tei_path = tmp_path / "doc.tei.xml"
    tei_path.write_text(
        """
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <div><head>Objective</head><p>To describe disease incidence in a population sample.</p></div>
              <div><head>Methods</head><p>We extracted surveillance records and calculated rates by group
              using a repeatable field protocol across all included sites.</p></div>
              <div><head>Results</head><p>Adults had the highest incidence and one subgroup had higher risk
              during the observed period.</p></div>
              <div><head>Conclusions</head><p>The evidence supports targeted surveillance and prevention
              while keeping the later introduction outside the abstract.</p></div>
              <div><head>Introduction</head><p>This must not bleed into the abstract.</p></div>
            </body>
          </text>
        </TEI>
        """,
        encoding="utf-8",
    )

    assert extract_tei_abstracts(tei_path) == [
        "Objective To describe disease incidence in a population sample. "
        "Methods We extracted surveillance records and calculated rates by group "
        "using a repeatable field protocol across all included sites. "
        "Results Adults had the highest incidence and one subgroup had higher risk "
        "during the observed period. "
        "Conclusions The evidence supports targeted surveillance and prevention "
        "while keeping the later introduction outside the abstract."
    ]


def test_extract_tei_abstracts_carries_inline_summary_into_next_structured_language(tmp_path: Path) -> None:
    tei_path = tmp_path / "doc.tei.xml"
    tei_path.write_text(
        """
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <div>
                <head>Objetivo</head>
                <p>Describir la incidencia en adultos. Palabras claves: salud, vigilancia. Summary
                Initial reports identified risk differences across groups and motivated a separate
                analysis of middle-aged adults in the surveillance data with enough detail to form
                the beginning of the English abstract.</p>
              </div>
              <div><head>Objective</head>
                <p>To describe disease incidence in a population sample using official reports.</p>
              </div>
              <div><head>Results</head>
                <p>Adults had the highest incidence and one subgroup had higher risk.</p>
              </div>
              <div><head>Conclusions</head>
                <p>The evidence supports targeted surveillance and prevention for the observed population.</p>
              </div>
            </body>
          </text>
        </TEI>
        """,
        encoding="utf-8",
    )

    abstracts = extract_tei_abstracts(tei_path)

    assert len(abstracts) == 1
    assert abstracts[0].startswith("Initial reports identified risk differences")
    assert "Objective To describe disease incidence" in abstracts[0]
    assert "Conclusions The evidence supports targeted surveillance" in abstracts[0]
