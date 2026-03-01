document.addEventListener('DOMContentLoaded', () => {
    // --- Element References ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const previewSection = document.getElementById('preview-section');
    const imagePreview = document.getElementById('image-preview');
    const imageWrapper = document.getElementById('image-wrapper');
    const filenameLabel = document.getElementById('filename-label');
    const analyzeBtn = document.getElementById('analyze-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    const resultsSection = document.getElementById('results-section');
    const resultHeading = document.getElementById('result-heading');
    const resultDescription = document.getElementById('result-description');
    const resultIcon = document.getElementById('result-icon');
    const confidenceContainer = document.getElementById('confidence-container');
    const confidenceValue = document.getElementById('confidence-value');
    const progressBar = document.getElementById('progress-bar');
    const lowConfidenceWarning = document.getElementById('low-confidence-warning');
    const resetBtn = document.getElementById('reset-btn');

    let currentFile = null;

    // ─── Drag & Drop ─────────────────────────────────────
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    // ─── Handle File Selection ────────────────────────────
    function handleFile(file) {
        const allowed = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!allowed.includes(file.type)) {
            showToast('Please upload a JPG or PNG image file.');
            return;
        }

        currentFile = file;
        const reader = new FileReader();

        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            filenameLabel.textContent = file.name;
            uploadSection.classList.add('hidden');
            previewSection.classList.remove('hidden');
        };

        reader.readAsDataURL(file);
    }

    cancelBtn.addEventListener('click', resetUI);

    // ─── Analyse Image ────────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI → Loading state
        analyzeBtn.disabled = true;
        cancelBtn.disabled = true;
        analyzeBtn.innerHTML = `
            <svg style="width:16px;height:16px;animation:spin 1s linear infinite;flex-shrink:0;" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-opacity="0.25"/>
                <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
            </svg>
            Analyzing…
        `;
        imageWrapper.classList.add('scanning');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server error ${response.status}`);
            }

            const data = await response.json();

            // Small delay so the scan animation is visible
            setTimeout(() => showResults(data), 1200);

        } catch (error) {
            console.error('Prediction error:', error);
            imageWrapper.classList.remove('scanning');
            analyzeBtn.disabled = false;
            cancelBtn.disabled = false;
            analyzeBtn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17H3a2 2 0 01-2-2V5a2 2 0 012-2h14a2 2 0 012 2v10a2 2 0 01-2 2h-2"/></svg>
                Analyze Image
            `;
            showToast(`Analysis failed: ${error.message}`);
        }
    });

    // ─── Show Results ─────────────────────────────────────
    function showResults(data) {
        imageWrapper.classList.remove('scanning');
        previewSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        lowConfidenceWarning.classList.add('hidden');
        resultIcon.className = 'result-icon';

        if (data.is_cancer) {
            resultHeading.textContent = '⚠ Cancer Detected';
            resultHeading.className = 'result-positive';
            resultDescription.innerHTML = `Predicted type: <strong>${data.skin_type}</strong>`;
            progressBar.className = 'progress-bar-fill bar-danger';
            resultIcon.classList.add('danger');
            resultIcon.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>`;
        } else {
            resultHeading.textContent = '✓ Non-Cancerous';
            resultHeading.className = 'result-negative';
            resultDescription.innerHTML = `Detected type: <strong>${data.skin_type}</strong>`;
            resultIcon.classList.add('success');
            resultIcon.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`;

            if (data.confidence > 80) progressBar.className = 'progress-bar-fill bar-high';
            else if (data.confidence > 50) progressBar.className = 'progress-bar-fill bar-medium';
            else progressBar.className = 'progress-bar-fill bar-low';
        }

        // Show & animate confidence bar
        confidenceContainer.classList.remove('hidden');
        requestAnimationFrame(() => {
            setTimeout(() => {
                progressBar.style.width = `${data.confidence}%`;
                animateCounter(confidenceValue, 0, data.confidence, 1000, (val) => `${val.toFixed(1)}%`);
            }, 80);
        });

        if (data.low_confidence) {
            lowConfidenceWarning.classList.remove('hidden');
        }
    }

    // ─── Counter Animation ────────────────────────────────
    function animateCounter(el, from, to, duration, format) {
        let start = null;
        const step = (ts) => {
            if (!start) start = ts;
            const progress = Math.min((ts - start) / duration, 1);
            el.textContent = format(from + (to - from) * easeOut(progress));
            if (progress < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    }

    function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

    // ─── Reset ────────────────────────────────────────────
    function resetUI() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        filenameLabel.textContent = '';

        uploadSection.classList.remove('hidden');
        previewSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        confidenceContainer.classList.add('hidden');

        imageWrapper.classList.remove('scanning');
        analyzeBtn.disabled = false;
        cancelBtn.disabled = false;
        analyzeBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17H3a2 2 0 01-2-2V5a2 2 0 012-2h14a2 2 0 012 2v10a2 2 0 01-2 2h-2"/></svg>
            Analyze Image
        `;
        progressBar.style.width = '0%';
        confidenceValue.textContent = '0%';
        lowConfidenceWarning.classList.add('hidden');
    }

    resetBtn.addEventListener('click', resetUI);

    // ─── Toast Notification ───────────────────────────────
    function showToast(msg) {
        const existing = document.getElementById('toast-container');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.id = 'toast-container';
        toast.textContent = msg;
        Object.assign(toast.style, {
            position: 'fixed',
            bottom: '1.5rem',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(239,68,68,0.15)',
            border: '1px solid rgba(239,68,68,0.3)',
            color: '#fca5a5',
            padding: '0.75rem 1.25rem',
            borderRadius: '10px',
            fontSize: '0.875rem',
            fontFamily: 'Inter, sans-serif',
            backdropFilter: 'blur(10px)',
            zIndex: '9999',
            maxWidth: '90vw',
            textAlign: 'center',
            fontWeight: '500',
            animation: 'fadeUp 0.3s ease',
        });
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }
});
