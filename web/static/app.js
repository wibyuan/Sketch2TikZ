/**
 * Sketch2TikZ Web UI — Frontend Controller
 */

// ── State ───────────────────────────────────────
let currentTaskId = null;
let currentImageFile = null;
let abortController = null;

// ── DOM refs ────────────────────────────────────
const $ = (id) => document.getElementById(id);

const uploadZone = $('upload-zone');
const fileInput = $('file-input');
const uploadPreview = $('upload-preview');
const previewImg = $('preview-img');
const btnRemoveFile = $('btn-remove-file');
const customPrompt = $('custom-prompt');
const btnGenerate = $('btn-generate');
const btnText = btnGenerate.querySelector('.btn-text');
const btnSpin = btnGenerate.querySelector('.btn-icon-spin');

const progressArea = $('progress-area');
const progressLabel = $('progress-label');
const progressPercent = $('progress-percent');
const progressFill = $('progress-fill');
const progressMessage = $('progress-message');

const emptyState = $('empty-state');
const resultView = $('result-view');
const resultImg = $('result-img');
const scoreBar = $('score-bar');
const scoreValue = $('score-value');
const scoreDiagnosis = $('score-diagnosis');
const scoreTime = $('score-time');
const codeEditor = $('code-editor');

const btnDlPdf = $('btn-dl-pdf');
const btnDlPng = $('btn-dl-png');
const btnCopyCode = $('btn-copy-code');
const btnDlTex = $('btn-dl-tex');
const btnRegenerate = $('btn-regenerate');
const btnCompileOnly = $('btn-compile-only');

const btnHistory = $('btn-history');
const historyPanel = $('history-panel');
const btnCloseHistory = $('btn-close-history');
const historyList = $('history-list');

const toast = $('toast');
const toastMsg = $('toast-msg');

// ── Init ────────────────────────────────────────
function init() {
    // Upload interactions
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileSelect(e.target.files[0]);
    });
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length) handleFileSelect(files[0]);
    });
    btnRemoveFile.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    // Generate
    btnGenerate.addEventListener('click', startGeneration);

    // Downloads
    btnDlPdf.addEventListener('click', () => downloadFile('pdf'));
    btnDlPng.addEventListener('click', () => downloadFile('png'));
    btnDlTex.addEventListener('click', () => downloadFile('tex'));
    btnCopyCode.addEventListener('click', copyCode);

    // Regenerate & compile
    btnRegenerate.addEventListener('click', startRegeneration);
    btnCompileOnly.addEventListener('click', compileOnly);

    // History panel
    btnHistory.addEventListener('click', () => historyPanel.classList.remove('hidden'));
    btnCloseHistory.addEventListener('click', () => historyPanel.classList.add('hidden'));

    // Load history
    loadHistory();
}

// ── File Handling ───────────────────────────────
function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showToast('请上传图片文件');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showToast('文件过大，最大支持 10MB');
        return;
    }
    currentImageFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    uploadPreview.classList.remove('hidden');
    uploadZone.querySelector('.upload-placeholder').classList.add('hidden');
    btnGenerate.disabled = false;
}

function clearFile() {
    currentImageFile = null;
    fileInput.value = '';
    previewImg.src = '';
    uploadPreview.classList.add('hidden');
    uploadZone.querySelector('.upload-placeholder').classList.remove('hidden');
    btnGenerate.disabled = true;
}

// ── Generation ──────────────────────────────────
async function startGeneration() {
    if (!currentImageFile) return;

    // UI: switch to generating state
    setGenerating(true);
    resetProgress();
    progressArea.classList.remove('hidden');
    emptyState.classList.add('hidden');
    resultView.classList.add('hidden');
    scoreBar.classList.add('hidden');

    const form = new FormData();
    form.append('file', currentImageFile);
    const promptVal = customPrompt.value.trim();
    if (promptVal) form.append('custom_prompt', promptVal);

    try {
        const res = await fetch('/api/generate', { method: 'POST', body: form });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const { task_id } = await res.json();
        currentTaskId = task_id;
        connectSSE(task_id);
    } catch (e) {
        showToast('生成失败: ' + e.message);
        setGenerating(false);
        progressLabel.textContent = '生成失败';
        progressFill.style.width = '100%';
        progressFill.style.background = 'var(--error)';
    }
}

function connectSSE(taskId) {
    if (abortController) abortController.abort();
    abortController = new AbortController();

    const evtSrc = new EventSource(`/api/tasks/${taskId}/stream`);

    evtSrc.onmessage = (e) => {
        const data = JSON.parse(e.data);
        handleEvent(data);
        if (data.stage === 'close') {
            evtSrc.close();
        }
    };

    evtSrc.onerror = () => {
        evtSrc.close();
    };
}

function handleEvent(ev) {
    const { stage, status, message, data } = ev;

    // Progress bar mapping
    const progMap = {
        vision: { pct: 15, label: '图像分析' },
        codegen: { pct: 40, label: '代码生成' },
        compile: { pct: 65, label: '编译渲染' },
        critic: { pct: 85, label: '质量评审' },
        done: { pct: 100, label: '完成' },
        error: { pct: 100, label: '出错' },
    };
    const info = progMap[stage] || { pct: 0, label: stage };

    progressLabel.textContent = info.label;
    progressPercent.textContent = `${info.pct}%`;
    progressFill.style.width = `${info.pct}%`;
    progressMessage.textContent = message;

    if (status === 'fail' || status === 'retry') {
        progressFill.style.background = status === 'fail' ? 'var(--error)' : 'var(--warning)';
    } else {
        progressFill.style.background = 'linear-gradient(90deg, var(--accent), var(--accent-hover))';
    }

    if (stage === 'done') {
        setGenerating(false);
        showResult(data);
        loadHistory();
    } else if (stage === 'error') {
        setGenerating(false);
        progressLabel.textContent = '生成失败';
        progressPercent.textContent = '!';
        showToast('生成失败: ' + message);
    }
}

function resetProgress() {
    progressLabel.textContent = '准备中...';
    progressPercent.textContent = '0%';
    progressFill.style.width = '0%';
    progressFill.style.background = 'linear-gradient(90deg, var(--accent), var(--accent-hover))';
    progressMessage.textContent = '';
}

function setGenerating(isGen) {
    btnGenerate.disabled = isGen || !currentImageFile;
    btnText.textContent = isGen ? '生成中...' : '开始生成';
    btnSpin.classList.toggle('hidden', !isGen);
}

// ── Result Display ──────────────────────────────
function showResult(result) {
    resultView.classList.remove('hidden');

    if (result.compile_ok) {
        // Load rendered PNG preview
        resultImg.src = `/api/tasks/${result.task_id}/preview?t=${Date.now()}`;

        // Score
        scoreBar.classList.remove('hidden');
        scoreValue.textContent = (result.critic_final_score || result.critic_first_score || 0).toFixed(1) + ' / 5.0';
        scoreDiagnosis.textContent = result.diagnosis || '无诊断信息';
        scoreTime.textContent = (result.total_time || 0) + 's';

        // Code
        codeEditor.value = result.tikz_code || '';
    } else {
        scoreBar.classList.remove('hidden');
        scoreValue.textContent = '—';
        scoreValue.style.color = 'var(--error)';
        scoreDiagnosis.textContent = result.diagnosis || '编译失败';
        scoreTime.textContent = (result.total_time || 0) + 's';
        codeEditor.value = result.tikz_code || '';
        resultImg.src = `/api/tasks/${result.task_id}/preview?t=${Date.now()}`;
    }
}

// ── Direct Compile (no AI) ─────────────────────
async function compileOnly() {
    if (!currentTaskId) return;
    const code = codeEditor.value.trim();
    if (!code) {
        showToast('代码为空，无法编译');
        return;
    }

    btnCompileOnly.disabled = true;
    btnCompileOnly.querySelector('span, svg')?.nextSibling?.textContent
        ? null
        : (btnCompileOnly.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon-spin"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> 编译中...');

    try {
        const form = new URLSearchParams();
        form.append('task_id', currentTaskId);
        form.append('code', code);
        const res = await fetch('/api/compile', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: form.toString()
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            showToast('编译失败: ' + (err.detail?.errors || err.detail || 'Unknown error'));
        } else {
            const data = await res.json();
            resultImg.src = data.preview_url;
            showToast('编译成功！预览已更新');
        }
    } catch (e) {
        showToast('编译请求失败: ' + e.message);
    } finally {
        btnCompileOnly.disabled = false;
        btnCompileOnly.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg> 直接编译修改后的代码';
    }
}

// ── Regeneration (with AI) ──────────────────────
async function startRegeneration() {
    if (!currentTaskId || !currentImageFile) return;

    const promptVal = customPrompt.value.trim();
    if (!promptVal) {
        showToast('请在左侧「自定义描述」中输入修改后的描述，再点击重新生成');
        customPrompt.focus();
        return;
    }

    setGenerating(true);
    resetProgress();
    progressArea.classList.remove('hidden');
    emptyState.classList.add('hidden');
    resultView.classList.add('hidden');

    const form = new FormData();
    form.append('file', currentImageFile);
    form.append('custom_prompt', promptVal);

    try {
        const res = await fetch('/api/generate', { method: 'POST', body: form });
        const { task_id } = await res.json();
        currentTaskId = task_id;
        connectSSE(task_id);
    } catch (e) {
        showToast('重新生成失败: ' + e.message);
        setGenerating(false);
    }
}

// ── Downloads ───────────────────────────────────
function downloadFile(fmt) {
    if (!currentTaskId) return;
    const a = document.createElement('a');
    a.href = `/api/download/${currentTaskId}/${fmt}`;
    a.download = `${currentTaskId}_${fmt}`;
    a.click();
}

async function copyCode() {
    const code = codeEditor.value;
    if (!code) return;
    try {
        await navigator.clipboard.writeText(code);
        showToast('代码已复制到剪贴板');
    } catch {
        showToast('复制失败，请手动复制');
    }
}

// ── History ─────────────────────────────────────
async function loadHistory() {
    try {
        const res = await fetch('/api/history');
        const items = await res.json();
        renderHistory(items);
    } catch {
        // silent fail
    }
}

function renderHistory(items) {
    if (!items.length) {
        historyList.innerHTML = '<div class="history-empty">暂无历史记录</div>';
        return;
    }
    // Sort by time descending
    items.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

    historyList.innerHTML = items.map(item => {
        const date = item.created_at ? new Date(item.created_at * 1000).toLocaleString('zh-CN', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' }) : '';
        const scoreText = item.score ? `${item.score.toFixed(1)}/5.0` : (item.compile_ok ? '编译通过' : '失败');
        const thumb = `/api/tasks/${item.task_id}/preview`;
        return `
            <div class="history-item" data-task-id="${item.task_id}">
                <img src="${thumb}" loading="lazy" onerror="this.style.opacity=0.3">
                <div class="history-item-info">
                    <div class="time">${date}</div>
                    <div class="score">${scoreText}${item.custom_prompt ? ' · 自定义' : ''}</div>
                </div>
            </div>
        `;
    }).join('');

    // Click to restore
    historyList.querySelectorAll('.history-item').forEach(el => {
        el.addEventListener('click', () => restoreTask(el.dataset.taskId));
    });
}

async function restoreTask(taskId) {
    try {
        const res = await fetch(`/api/tasks/${taskId}`);
        const task = await res.json();
        if (!task.result) {
            showToast('该任务暂无结果');
            return;
        }
        currentTaskId = taskId;
        showResult(task.result);
        emptyState.classList.add('hidden');
        historyPanel.classList.add('hidden');
    } catch {
        showToast('加载历史记录失败');
    }
}

// ── Toast ───────────────────────────────────────
let toastTimer = null;
function showToast(msg) {
    toastMsg.textContent = msg;
    toast.classList.remove('hidden');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.add('hidden'), 3000);
}

// ── Boot ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
