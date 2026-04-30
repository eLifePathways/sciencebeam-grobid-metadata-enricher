HEADER_METADATA_PROMPT = (
    "Extract structured header metadata from the provided PDF lines. "
    "Use only the supplied lines. "
    "Return JSON with fields: title, authors, affiliations, abstract, keywords. "
    "Use empty strings or empty lists when a field is not present."
)

TEI_METADATA_PROMPT = (
    "Extract structured metadata from the provided GROBID TEI header snippet. "
    "Use only the supplied TEI. "
    "Return JSON with fields: title, authors, affiliations, abstract, keywords, "
    "publisher, date, language, identifiers, relations, rights, types, formats. "
    "Use empty strings or empty lists when a field is not present.\n"
    "The 'identifiers' field is the persistent identifier(s) OF THIS ARTICLE only "
    "(its DOI, PMID, PMCID, arXiv ID, or article URL). Do NOT include author ORCID "
    "iDs, reviewer iDs, ISSN of the journal, ISBN of a book, or DOIs of cited "
    "references — those belong to other entities, not this article."
)

ABSTRACT_SELECTION_PROMPT = (
    "Select the true scientific abstract from the candidate text blocks. "
    "Ignore deposit or disclaimer text. "
    "If multiple abstracts appear in different languages, prefer English. "
    'Return JSON only: {"abstract": "..."} or {"abstract": ""}.'
)

OCR_CLEANUP_PROMPT = (
    "Clean noisy PDF or OCR text. "
    "Fix broken words, hyphenation, and line breaks. "
    "Remove obvious repeated headers or footers. "
    "Do not add content. "
    "Return plain text only."
)

ABSTRACT_EXTRACTION_PROMPT = (
    "Extract the scientific abstract from the corrected text. "
    "Ignore deposit or disclaimer text. "
    "If multiple abstracts appear in different languages, prefer English. "
    'Return JSON only: {"abstract": "..."} or {"abstract": ""}.'
)

KEYWORD_TRANSLATION_PROMPT = (
    "Translate the provided keyword list into the requested target languages. "
    "Preserve technical terms and proper names. "
    'Return JSON only: {"translations": {"en": [...], "pt": [...], "es": [...]}}. '
    "Only include requested languages."
)

CONTENT_EXTRACTION_PROMPT = (
    "Extract the structured content of this scientific article from the PDF layout text below. "
    "Use ONLY the supplied text; do not invent anything. "
    "Return JSON with these fields: "
    "body_sections (list of section heading strings in reading order, e.g. 'Introduction', 'Methods', 'Results'); "
    "figure_captions (list of figure caption strings, one per figure, including the 'Figure N.' label); "
    "table_captions (list of table caption strings, one per table, including the 'Table N.' label); "
    "reference_titles (list of article titles cited in the reference list, one per reference). "
    "Omit any field that the text does not support. Keep each list ordered as items appear in the document. "
    'Return JSON only: '
    '{"body_sections": [...], "figure_captions": [...], "table_captions": [...], "reference_titles": [...]}.'
)

REFERENCES_EXTRACTION_PROMPT = (
    "Extract every bibliographic reference listed in the text below. "
    "The text is the reference list or end-of-document text of a scientific paper. "
    "For each distinct numbered or unnumbered reference, return the article or chapter or book title as a string. "
    "Do not include author names, journal, year, or page numbers. Return titles only. "
    "Preserve the order the references appear. "
    "Be exhaustive; there may be 30 or more references and all of them should be returned. "
    "If you see a DOI next to a reference, also include it. "
    "Do not invent; only return what the text contains. "
    'Return JSON only: {"references": [{"title": "...", "doi": "..."}]}.'
)

TABLES_FIGURES_EXTRACTION_PROMPT = (
    "Extract every table caption and figure caption from the scientific paper text below. "
    "The text is the PDF layout and may contain headers, footers, body paragraphs, "
    "figure captions, table captions, and table cells. "
    "A table caption usually begins with 'Table N' or 'Tabla N' or 'Tabela N' where N is a number. "
    "A figure caption usually begins with 'Figure N', 'Fig. N', or 'Figura N'. "
    "Be exhaustive; a paper can contain 10 or more tables and 10 or more figures. "
    "Return the full caption text (label plus description), in document order. "
    "Do not include table cell content, only the caption that describes the table or figure. "
    "Do not invent captions; only return what the text contains. "
    'Return JSON only: {"tables": ["Table 1. ...", ...], "figures": ["Figure 1. ...", ...]}.'
)
