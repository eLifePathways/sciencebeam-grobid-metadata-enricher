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

IDENTIFIER_SELECTION_PROMPT = (
    "Select the persistent identifiers (DOIs, PMIDs, PMCIDs, arXiv IDs, article URLs) "
    "of THIS scientific article from the supplied candidate identifier lists and front-matter lines. "
    "Use the title and the record_id as additional context to pick the article's own identifiers, "
    "not those of cited references, journals, books, authors, or reviewers. "
    "Treat record_id-derived candidates as lower-confidence fallback identifiers. "
    "Reject author ORCIDs, reviewer IDs, journal ISSN, book ISBN, grant IDs, hash-like values, "
    "and DOIs or URLs that belong to cited references or other articles. "
    'Return JSON only: {"identifiers": ["..."]}.'
)

BODY_SECTIONS_EXTRACTION_PROMPT = (
    "Select the article's JATS-style body section titles from the supplied ALTO candidate lines. "
    "The candidates are already layout-filtered and begin at the body of the paper. "
    "Be exhaustive: keep main sections, subsection headings, method subheadings, and unnumbered headings "
    "in document order. Do not collapse the list to only Introduction/Methods/Results/Discussion. "
    "Select headings from the author's narrative section hierarchy, not every bold or separated line. "
    "When the candidate list is noisy, prefer candidates that fit a coherent numbered hierarchy, repeated "
    "heading style, or an unnumbered heading followed by paragraph text. "
    "Keep numbered heading prefixes exactly when they appear, such as '1 Introduction' or '2.1 Methods'; "
    "omit only standalone numbering with no heading text. "
    "Keep body-level administrative or availability sections such as Plain language summary, Ethics and consent, "
    "Data availability, Underlying data, Extended data, and Supporting information when they appear as section titles. "
    "Use only candidate line text; do not invent headings. "
    "Reject article title, authors, affiliations, abstract/keywords labels, page headers or footers, "
    "figure captions, figure panel labels, table captions, table cells, reference list entries, "
    "numbered procedure/checklist items, table row or column labels, glossary/list labels, "
    "reviewer-report text, statistical-result labels, isolated emphasized phrases inside paragraphs, "
    "and prose sentences that are not headings. "
    "Use following_text only to decide whether the candidate starts a section; "
    "do not copy following_text as a heading. "
    "Copy the selected heading text exactly as it appears after 'candidate:', except omit pure numbering only. "
    'Return JSON only: {"body_sections": ["Introduction", "..."]}.'
)

REFERENCES_EXTRACTION_PROMPT = (
    "Extract every bibliographic reference listed in the supplied ALTO reference candidate entries. "
    "The candidates are already bounded to the reference-list region of a scientific paper. "
    "For each distinct numbered or unnumbered reference, return the article or chapter or book title as a string. "
    "Do not include author names, journal, year, or page numbers. Return titles only. "
    "Preserve the order the references appear. "
    "Be exhaustive; there may be 30 or more references and all of them should be returned. "
    "If you see a DOI next to a reference, also include it. "
    "Do not invent; only return titles and DOIs supported by the supplied candidate entries. "
    'Return JSON only: {"references": [{"title": "...", "doi": "..."}]}.'
)

FIGURE_CAPTIONS_SELECTION_PROMPT = (
    "Select the true figure captions from the supplied ALTO candidate list. "
    "Each candidate starts with a figure label such as Figure 1, Fig. 1, Figura 1, or Esquema 1, "
    "and may include continuation lines joined from the PDF layout. "
    "Keep only captions that describe an actual figure, panel, diagram, scheme, or supplementary figure. "
    "Reject in-text figure references, standalone panel labels, body prose, table captions, table cells, "
    "page headers, and footers. "
    "Return the full caption text exactly from the candidate after the separator, preserving the figure label. "
    "Be exhaustive and keep document order. "
    'Return JSON only: {"figures": ["Figure 1. ...", "Fig. S1. ..."]}.'
)

TABLE_CAPTIONS_SELECTION_PROMPT = (
    "Select the true table captions from the supplied ALTO candidate list. "
    "Each candidate starts with a table label such as Table 1, Table S1, Tabla 1, or Tabela 1, "
    "and may include continuation lines joined from the PDF layout. "
    "Keep only captions that describe an actual table in the article or supplementary material. "
    "Keep pseudocode, parameter, dataset, antibody, siRNA, and supplementary-table captions when they are "
    "introduced by a table label and describe the table. "
    "Reject in-text table references, reviewer comments, standalone table cells, column headers, and body prose. "
    "Return the full caption text exactly from the candidate after the separator, preserving the table label. "
    "Be exhaustive and keep document order. "
    'Return JSON only: {"tables": ["Table 1. ...", "Table S1. ..."]}.'
)

# TODO(rewrite): placeholder.
KEYWORD_EXTRACTION_PROMPT = (
    "Extract the article's author-supplied keyword lists from the front-matter lines. "
    "Different language sections may carry their own keyword list (English Keywords, Portuguese Palavras-chave, "
    "Spanish Palabras clave, Descritores, Descriptors). "
    "Return one entry per language list, with the canonical language code (en, pt, es). "
    "Use only the supplied front-matter text; do not infer keywords from the title or abstract. "
    "If no explicit keyword list is present, return an empty list. "
    'Return JSON only: {"keyword_lists": [{"language": "en", "keywords": ["...", "..."]}]}.'
)

# TODO(rewrite): placeholder.
KEYWORD_INFERENCE_PROMPT = (
    "Infer 5-8 plausible keywords for this scientific article from its title and selected abstract. "
    "Each keyword should be a 1-8 word noun phrase representing a key concept, method, or topic. "
    "Avoid stopword-only phrases, numeric values, and overly broad terms. "
    "Use only what the supplied title and abstract justify; do not invent results that aren't grounded. "
    'Return JSON only: {"keywords": ["...", "..."]}.'
)

# TODO(rewrite): placeholder.
KEYWORD_SELECTION_PROMPT = (
    "Select the keyword list that matches the article's primary language and content. "
    "Multiple candidate lists may have been extracted from different parts of the front matter or "
    "different language sections. Prefer the list whose language matches the requested language code, "
    "that contains real domain-specific terms (not journal labels or affiliations), and that aligns "
    "with the title and abstract. "
    "Return ONLY entries that appear verbatim in one of the candidate_keyword_lists; do not invent "
    "or merge across lists. "
    'Return JSON only: {"keywords": ["...", "..."]}.'
)
