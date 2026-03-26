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
    "Use empty strings or empty lists when a field is not present."
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
