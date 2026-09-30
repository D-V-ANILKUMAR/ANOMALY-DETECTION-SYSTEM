/**
 * Anomaly Detection System — Frontend Application
 * 
 * Handles image upload, detection API calls, visualization switching,
 * threshold adjustment, and PDF report download.
 */

// ─── State ──────────────────────────────────────────────────────────────────
const state = {
    selectedFile: null,
    resultData: null,
    currentView: 'original',
    isProcessing: false,
};

// ─── DOM Elements ───────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const uploadZone = $('#uploadZone');
const fileInput = $('#fileInput');
const fileInfo = $('#fileInfo');
const fileName = $('#fileName');
const fileRemove = $('#fileRemove');
const imageType = $('#imageType');
const detectionMethod = $('#detectionMethod');
const methodGroup = $('#methodGroup');
const methodInfoText = $('#methodInfoText');
const thresholdSlider = $('#thresholdSlider');
const thresholdValue = $('#thresholdValue');
const detectBtn = $('#detectBtn');
const emptyState = $('#emptyState');
const resultsContainer = $('#resultsContainer');
const processingOverlay = $('#processingOverlay');
const processingText = $('#processingText');
const toastContainer = $('#toastContainer');
const quickStatsPanel = $('#quickStatsPanel');

// ─── Toast Notifications ────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
    
    toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.transition = 'all 0.3s ease';
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ─── File Upload ────────────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileSelect(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
});

fileRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
});

function handleFileSelect(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const validExtensions = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp', 'npy', 'hdr', 'mat'];
    
    if (!validExtensions.includes(ext)) {
        showToast('Unsupported file format. Use PNG, JPG, TIFF, NPY, or HDR.', 'error');
        return;
    }
    
    if (file.size > 50 * 1024 * 1024) {
        showToast('File too large. Maximum size is 50MB.', 'error');
        return;
    }
    
    state.selectedFile = file;
    uploadZone.classList.add('has-file');
    fileName.textContent = file.name;
    fileInfo.classList.add('visible');
    detectBtn.disabled = false;
    
    // Auto-detect image type
    const hyperExtensions = ['npy', 'hdr', 'mat'];
    if (hyperExtensions.includes(ext)) {
        imageType.value = 'hyperspectral';
        updateMethodVisibility();
    } else {
        imageType.value = 'rgb';
        updateMethodVisibility();
    }
    
    showToast(`File loaded: ${file.name}`, 'success');
}

function clearFile() {
    state.selectedFile = null;
    fileInput.value = '';
    uploadZone.classList.remove('has-file');
    fileInfo.classList.remove('visible');
    detectBtn.disabled = true;
}

// ─── Settings ───────────────────────────────────────────────────────────────
imageType.addEventListener('change', updateMethodVisibility);
detectionMethod.addEventListener('change', updateMethodInfo);

thresholdSlider.addEventListener('input', (e) => {
    thresholdValue.textContent = `${e.target.value}%`;
});

function updateMethodVisibility() {
    if (imageType.value === 'hyperspectral') {
        methodGroup.style.display = 'none';
    } else {
        methodGroup.style.display = 'block';
    }
    updateMethodInfo();
}

function updateMethodInfo() {
    const descriptions = {
        mahalanobis: 'Uses robust covariance estimation (MCD) to compute Mahalanobis distance. Best for detecting pixels that deviate from the global distribution.',
        statistical: 'Computes Z-scores across multiple feature dimensions. Effective for detecting extreme outliers in color, texture, and gradient features.',
        rx: 'RX (Reed-Xiaoli) detector computes spectral Mahalanobis distance. Standard method for hyperspectral anomaly detection.'
    };
    
    if (imageType.value === 'hyperspectral') {
        methodInfoText.textContent = descriptions.rx;
        $('#methodInfo').style.display = 'flex';
        methodGroup.style.display = 'none';
    } else {
        methodInfoText.textContent = descriptions[detectionMethod.value] || descriptions.mahalanobis;
        $('#methodInfo').style.display = 'flex';
    }
}

// ─── Progress Animation ─────────────────────────────────────────────────────
function setProgressStep(stepName) {
    const steps = $$('.progress-step');
    let found = false;
    
    steps.forEach(step => {
        const name = step.dataset.step;
        if (name === stepName) {
            step.classList.add('active');
            step.classList.remove('done');
            found = true;
        } else if (!found) {
            step.classList.remove('active');
            step.classList.add('done');
        } else {
            step.classList.remove('active', 'done');
        }
    });
}

function resetProgress() {
    $$('.progress-step').forEach(step => {
        step.classList.remove('active', 'done');
    });
}

// ─── Detection ──────────────────────────────────────────────────────────────
detectBtn.addEventListener('click', runDetection);

async function runDetection() {
    if (!state.selectedFile || state.isProcessing) return;
    
    state.isProcessing = true;
    detectBtn.disabled = true;
    detectBtn.classList.add('btn-loading');
    processingOverlay.classList.add('active');
    resetProgress();
    
    try {
        // Step 1: Upload
        setProgressStep('upload');
        processingText.textContent = 'Uploading image...';
        await sleep(400);
        
        // Step 2: Preprocess
        setProgressStep('preprocess');
        processingText.textContent = 'Extracting features...';
        
        const formData = new FormData();
        formData.append('image', state.selectedFile);
        formData.append('image_type', imageType.value);
        formData.append('method', detectionMethod.value);
        formData.append('threshold', thresholdSlider.value);
        
        // Step 3: Detect
        setProgressStep('detect');
        processingText.textContent = 'Running anomaly detection...';
        
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData,
        });
        
        // Read body text once (can only be consumed once per response)
        const responseText = await response.text();
        let data;
        try {
            data = JSON.parse(responseText);
        } catch (_) {
            // Server returned HTML (Render wake-up page / 502 / 504)
            if (!response.ok) {
                throw new Error(
                    `Server returned HTTP ${response.status}. ` +
                    `If this is the first request after a while, Render free tier may be waking up. ` +
                    `Please wait 30 seconds and try again.`
                );
            }
            throw new Error('Server returned an unexpected response. Please try again.');
        }
        
        if (!response.ok) {
            throw new Error(data.error || `Detection failed (HTTP ${response.status})`);
        }
        
        // Step 5: Complete
        setProgressStep('complete');
        processingText.textContent = 'Preparing results...';
        await sleep(400);
        
        displayResults(data);
        showToast('Anomaly detection complete!', 'success');
        
    } catch (error) {
        console.error('Detection error:', error);
        showToast(`Detection failed: ${error.message}`, 'error', 6000);
    } finally {
        state.isProcessing = false;
        detectBtn.disabled = false;
        detectBtn.classList.remove('btn-loading');
        processingOverlay.classList.remove('active');
        resetProgress();
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── Display Results ────────────────────────────────────────────────────────
function displayResults(data) {
    emptyState.style.display = 'none';
    resultsContainer.style.display = 'flex';
    resultsContainer.style.flexDirection = 'column';
    resultsContainer.style.gap = '24px';
    
    // Set images
    setImage('imgOriginal', data.original_base64);
    setImage('imgHeatmap', data.heatmap_base64);
    setImage('imgOverlay', data.overlay_base64);
    setImage('imgHighlighted', data.highlighted_base64);
    setImage('imgMask', data.mask_base64);
    
    // Switch to heatmap view to show results
    switchView('heatmap');
    
    // Update statistics
    const stats = data.statistics;
    $('#statTotalPixels').textContent = stats.total_pixels.toLocaleString();
    $('#statAnomalyPixels').textContent = stats.anomaly_pixels.toLocaleString();
    $('#statAnomalyPct').textContent = `${stats.anomaly_percentage}%`;
    $('#statRegions').textContent = stats.num_regions;
    
    $('#statMinScore').textContent = stats.min_score.toFixed(4);
    $('#statMaxScore').textContent = stats.max_score.toFixed(4);
    $('#statMeanScore').textContent = stats.mean_score.toFixed(4);
    $('#statStdScore').textContent = stats.std_score.toFixed(4);
    $('#statThresholdVal').textContent = stats.threshold_value.toFixed(4);
    $('#statMethod').textContent = data.method || 'N/A';
    
    // Quick stats sidebar
    quickStatsPanel.style.display = 'block';
    $('#quickAnomalyPct').textContent = `${stats.anomaly_percentage}%`;
    $('#quickRegions').textContent = stats.num_regions;
    $('#quickDimensions').textContent = `${data.width} × ${data.height}`;
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setImage(id, base64) {
    const img = document.getElementById(id);
    if (base64) {
        img.src = `data:image/png;base64,${base64}`;
    }
}

// ─── Visualization Tabs ─────────────────────────────────────────────────────
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('viz-tab')) {
        const view = e.target.dataset.view;
        switchView(view);
    }
});

function switchView(view) {
    state.currentView = view;
    
    // Update tab active state
    $$('.viz-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === view);
    });
    
    // Update image visibility
    const imageMap = {
        original: 'imgOriginal',
        heatmap: 'imgHeatmap',
        overlay: 'imgOverlay',
        highlighted: 'imgHighlighted',
        mask: 'imgMask',
    };
    
    $$('.viz-image').forEach(img => img.classList.remove('active'));
    const targetImg = document.getElementById(imageMap[view]);
    if (targetImg) targetImg.classList.add('active');
}

// ─── Threshold Adjustment ───────────────────────────────────────────────────
$('#adjustThresholdBtn').addEventListener('click', async () => {
    if (!state.resultData) return;
    
    const resultId = state.resultData.result_id;
    const threshold = thresholdSlider.value;
    
    showToast('Re-applying threshold...', 'info', 2000);
    
    try {
        const response = await fetch('/api/rethreshold', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ result_id: resultId, threshold: threshold }),
        });
        
        const responseText = await response.text();
        let data;
        try {
            data = JSON.parse(responseText);
        } catch (_) {
            throw new Error('Server returned an unexpected response. Please try again.');
        }
        
        if (!response.ok) {
            throw new Error(data.error || 'Re-threshold failed');
        }
        

        // Update visualizations
        setImage('imgHeatmap', data.heatmap_base64);
        setImage('imgOverlay', data.overlay_base64);
        setImage('imgHighlighted', data.highlighted_base64);
        setImage('imgMask', data.mask_base64);
        
        // Update statistics
        const stats = data.statistics;
        $('#statAnomalyPixels').textContent = stats.anomaly_pixels.toLocaleString();
        $('#statAnomalyPct').textContent = `${stats.anomaly_percentage}%`;
        $('#statRegions').textContent = stats.num_regions;
        $('#statThresholdVal').textContent = stats.threshold_value.toFixed(4);
        
        // Update quick stats
        $('#quickAnomalyPct').textContent = `${stats.anomaly_percentage}%`;
        $('#quickRegions').textContent = stats.num_regions;
        
        // Update stored data statistics
        state.resultData.statistics = stats;
        state.resultData.heatmap_base64 = data.heatmap_base64;
        state.resultData.overlay_base64 = data.overlay_base64;
        state.resultData.highlighted_base64 = data.highlighted_base64;
        state.resultData.mask_base64 = data.mask_base64;
        
        showToast('Threshold updated successfully!', 'success');
        
    } catch (error) {
        console.error('Re-threshold error:', error);
        showToast(`Failed: ${error.message}`, 'error');
    }
});

// ─── Report Download ────────────────────────────────────────────────────────
$('#downloadReportBtn').addEventListener('click', async () => {
    if (!state.resultData) return;
    
    showToast('Generating PDF report...', 'info', 3000);
    
    try {
        const response = await fetch(`/api/report/${state.resultData.result_id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state.resultData),
        });
        
        if (!response.ok) {
            throw new Error('Report generation failed');
        }
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `anomaly_report_${state.resultData.result_id}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        
        showToast('Report downloaded successfully!', 'success');
        
    } catch (error) {
        console.error('Report error:', error);
        showToast(`Report generation failed: ${error.message}`, 'error');
    }
});

// ─── Keyboard Shortcuts ─────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to run detection
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!detectBtn.disabled) detectBtn.click();
    }
    
    // Arrow keys to switch views (when results are shown)
    if (state.resultData && !state.isProcessing) {
        const views = ['original', 'heatmap', 'overlay', 'highlighted', 'mask'];
        const currentIdx = views.indexOf(state.currentView);
        
        if (e.key === 'ArrowRight' && currentIdx < views.length - 1) {
            switchView(views[currentIdx + 1]);
        } else if (e.key === 'ArrowLeft' && currentIdx > 0) {
            switchView(views[currentIdx - 1]);
        }
    }
});

// ─── Initialize ─────────────────────────────────────────────────────────────
updateMethodVisibility();
console.log('🔬 Anomaly Detection System initialized');
