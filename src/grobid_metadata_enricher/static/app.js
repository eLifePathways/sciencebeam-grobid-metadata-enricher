'use strict';

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileNameEl = document.getElementById('file-name');
const dropError = document.getElementById('drop-error');
const submitBtn = document.getElementById('submit-btn');
const statusSection = document.getElementById('status-section');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');
const errorDismiss = document.getElementById('error-dismiss');
const resultsSection = document.getElementById('results-section');

let selectedFile = null;

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) onFileSelected(file);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) onFileSelected(fileInput.files[0]);
});

submitBtn.addEventListener('click', onSubmit);

errorDismiss.addEventListener('click', () => {
    errorSection.hidden = true;
    setUploadEnabled(true);
});

function onFileSelected(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        dropError.textContent = 'Please select a PDF file.';
        dropError.hidden = false;
        selectedFile = null;
        submitBtn.disabled = true;
        return;
    }
    dropError.hidden = true;
    selectedFile = file;
    fileNameEl.textContent = '📄 ' + file.name;
    fileNameEl.hidden = false;
    submitBtn.disabled = false;
}

async function onSubmit() {
    if (!selectedFile) return;

    setUploadEnabled(false);
    statusSection.hidden = false;
    errorSection.hidden = true;
    resultsSection.hidden = true;

    try {
        const data = await fetchTransform(selectedFile);
        populateResults(data);
        statusSection.hidden = true;
        resultsSection.hidden = false;
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
        statusSection.hidden = true;
        errorMessage.textContent = err.message;
        errorSection.hidden = false;
        setUploadEnabled(true);
    }
}

async function fetchTransform(file) {
    const formData = new FormData();
    formData.append('file', file);

    let response;
    try {
        response = await fetch('/api/transform', { method: 'POST', body: formData });
    } catch {
        throw new Error('Network error — check your connection and try again.');
    }

    if (!response.ok) {
        let detail = '';
        try {
            const json = await response.json();
            detail = json.detail || '';
        } catch { /* ignore parse errors */ }

        const knownMessages = {
            400: 'Invalid file — please upload a PDF.',
            500: 'Pipeline processing failed.',
            502: 'Processing service unavailable.',
            503: 'LLM backend not configured.',
        };
        const base = knownMessages[response.status] || 'An unexpected error occurred.';
        throw new Error(detail ? `${base} ${detail}` : `${base} (${response.status})`);
    }

    return response.json();
}

function populateResults(data) {
    document.getElementById('f-title').value = data.title || '';
    document.getElementById('f-abstract').value = data.abstract || '';
    document.getElementById('f-publisher').value = data.publisher || '';
    document.getElementById('f-date').value = data.date || '';
    document.getElementById('f-language').value = data.language || '';
    document.getElementById('f-rights').value = data.rights || '';

    populateList('f-authors', data.authors);
    populateList('f-affiliations', data.affiliations);
    populateList('f-keywords', data.keywords);
    populateList('f-identifiers', data.identifiers);
    populateList('f-relations', data.relations);
    populateList('f-types', data.types);
    populateList('f-formats', data.formats);
}

function populateList(fieldId, items) {
    const container = document.getElementById(fieldId);
    container.innerHTML = '';

    if (!items || items.length === 0) {
        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = '(none)';
        input.disabled = true;
        container.appendChild(input);
        return;
    }

    items.forEach((item) => {
        const input = document.createElement('input');
        input.type = 'text';
        input.value = item;
        input.readOnly = true;
        container.appendChild(input);
    });
}

function setUploadEnabled(enabled) {
    fileInput.disabled = !enabled;
    dropZone.classList.toggle('disabled', !enabled);
    submitBtn.disabled = !enabled || !selectedFile;
}
