from __future__ import annotations

from pathlib import Path

from grobid_metadata_enricher.formats import extract_oai_dc


def test_extract_oai_dc_ignores_placeholder_gold_values(tmp_path: Path) -> None:
    xml_path = tmp_path / "record.xml"
    xml_path.write_text(
        """
        <record xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>Example</dc:title>
          <dc:description>N/A</dc:description>
          <dc:description>Real abstract.</dc:description>
          <dc:subject xml:lang="en">N/A</dc:subject>
          <dc:subject xml:lang="en">COVID-19</dc:subject>
          <dc:identifier>N/A</dc:identifier>
          <dc:identifier>https://preprints.scielo.org/index.php/scielo/preprint/view/1</dc:identifier>
        </record>
        """,
        encoding="utf-8",
    )

    record = extract_oai_dc(xml_path)

    assert record["abstract"] == "Real abstract."
    assert record["abstracts"] == ["Real abstract."]
    assert record["keywords"] == ["COVID-19"]
    assert record["keywords_groups"] == {"en": ["COVID-19"]}
    assert record["identifiers"] == ["https://preprints.scielo.org/index.php/scielo/preprint/view/1"]
