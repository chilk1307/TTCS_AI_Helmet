let currentVideoId = null;
let ws = null;
let statusInterval = null;
let pendingMeta = null;

if (!localStorage.getItem('token')) {
    window.location.href = '/';
}

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
        document.getElementById('tab-' + tab.dataset.tab).style.display = 'block';
    });
});

// File selection
const fileInput = document.getElementById('videoFile');
const uploadArea = document.getElementById('uploadArea');

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        document.getElementById('fileName').textContent = fileInput.files[0].name;
        document.getElementById('btnUpload').disabled = false;
    }
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#2563eb';
});
uploadArea.addEventListener('dragleave', () => { uploadArea.style.borderColor = ''; });
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '';
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        document.getElementById('fileName').textContent = e.dataTransfer.files[0].name;
        document.getElementById('btnUpload').disabled = false;
    }
});

async function uploadVideo() {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const btn = document.getElementById('btnUpload');
    btn.disabled = true;
    btn.textContent = 'Đang tải...';

    try {
        const res = await fetch('/api/videos/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (res.ok) {
            currentVideoId = data.video_id;
            btn.textContent = 'Đã tải lên';
            document.getElementById('btnStart').style.display = 'block';
        } else {
            alert('Lỗi upload: ' + (data.detail || 'Unknown'));
            btn.disabled = false;
            btn.textContent = 'Tải lên';
        }
    } catch (err) {
        alert('Lỗi kết nối: ' + err.message);
        btn.disabled = false;
        btn.textContent = 'Tải lên';
    }
}

async function startAnalysis() {
    if (!currentVideoId) return;

    try {
        const res = await fetch(`/api/videos/${currentVideoId}/start`, { method: 'POST' });
        if (res.ok) {
            document.getElementById('btnStart').style.display = 'none';
            document.getElementById('btnStop').style.display = 'block';
            setSystemStatus('Đang phân tích', '#fef3c7', '#92400e');
            connectWebSocket();
            startStatusPolling();
        }
    } catch (err) {
        alert('Lỗi: ' + err.message);
    }
}

async function stopAnalysis() {
    if (!currentVideoId) return;
    try { await fetch(`/api/videos/${currentVideoId}/stop`, { method: 'POST' }); } catch (err) {}
    disconnectWebSocket();
    stopStatusPolling();
    document.getElementById('btnStop').style.display = 'none';
    document.getElementById('btnStart').style.display = 'block';
    setSystemStatus('Đã dừng', '#fee2e2', '#dc2626');
}

function setSystemStatus(text, bg, color) {
    const el = document.getElementById('systemStatus');
    el.textContent = text;
    el.style.background = bg;
    el.style.color = color;
}

// WebSocket: receives text (JSON meta) then binary (JPEG frame)
function connectWebSocket() {
    if (ws) ws.close();

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/stream/${currentVideoId}`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    const streamFrame = document.getElementById('streamFrame');

    ws.onopen = () => {
        document.getElementById('videoPlaceholder').style.display = 'none';
        streamFrame.style.display = 'block';
    };

    ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
            const msg = JSON.parse(event.data);

            if (msg.status === 'completed' && !msg.fps) {
                onPipelineComplete();
                return;
            }
            pendingMeta = msg;
        } else {
            // Binary JPEG frame
            const blob = new Blob([event.data], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            const oldSrc = streamFrame.src;
            streamFrame.src = url;
            if (oldSrc && oldSrc.startsWith('blob:')) URL.revokeObjectURL(oldSrc);

            if (pendingMeta) {
                document.getElementById('fpsDisplay').textContent = pendingMeta.fps + ' FPS';
                updateStatusFromMeta(pendingMeta);
                pendingMeta = null;
            }
        }
    };

    ws.onclose = () => {};
    ws.onerror = () => {};
}

function disconnectWebSocket() {
    if (ws) { ws.close(); ws = null; }
}

function updateStatusFromMeta(m) {
    setDot('dotReader', true);
    setDot('dotStage1', m.s1fps > 0);
    document.getElementById('valInputFps').textContent = '-';
    document.getElementById('valStage1Fps').textContent = m.s1fps + ' fps';
    document.getElementById('valViolationFps').textContent = '-';
    document.getElementById('valQueueSize').textContent = (m.fq || 0) + ' / ' + (m.vq || 0);
    document.getElementById('valViolationsCount').textContent = m.vc || 0;
}

// Status polling (fallback + extra info)
function startStatusPolling() {
    statusInterval = setInterval(async () => {
        if (!currentVideoId) return;
        try {
            const res = await fetch(`/api/status/${currentVideoId}`);
            const data = await res.json();
            updateFullStatus(data);
            if (data.status === 'completed') onPipelineComplete();
        } catch (err) {}
    }, 2000);
}

function stopStatusPolling() {
    if (statusInterval) { clearInterval(statusInterval); statusInterval = null; }
}

function updateFullStatus(data) {
    setDot('dotReader', data.input_fps > 0);
    setDot('dotStage1', data.stage1_fps > 0);
    setDot('dotViolation', data.violation_fps > 0);

    document.getElementById('valInputFps').textContent = data.input_fps + ' fps';
    document.getElementById('valStage1Fps').textContent = data.stage1_fps + ' fps';
    document.getElementById('valViolationFps').textContent = data.violation_fps + ' fps';
    document.getElementById('valQueueSize').textContent = (data.frame_queue_size || 0) + ' / ' + (data.violation_queue_size || 0);
    document.getElementById('valViolationsCount').textContent = data.violations_count || 0;
}

function setDot(id, active) {
    const dot = document.getElementById(id);
    if (active) dot.classList.add('active');
    else dot.classList.remove('active');
}

function onPipelineComplete() {
    stopStatusPolling();
    document.getElementById('btnStop').style.display = 'none';
    setSystemStatus('Hoàn thành', '#dcfce7', '#16a34a');
    refreshViolations();
}

// Violations
async function refreshViolations() {
    try {
        const url = currentVideoId ? `/api/violations?video_id=${currentVideoId}` : '/api/violations';
        const res = await fetch(url);
        const data = await res.json();
        renderViolations(data.violations);
    } catch (err) {}
}

function renderViolations(violations) {
    const tbody = document.getElementById('violationsBody');
    if (!violations || violations.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-row">Chưa có vi phạm nào</td></tr>';
        return;
    }

    tbody.innerHTML = violations.map(v => `
        <tr>
            <td><code>${v.id || '-'}</code></td>
            <td>${v.ten || '-'}</td>
            <td><strong>${v.bien_so || '-'}</strong></td>
            <td>${v.thoi_gian || '-'}</td>
            <td>${v.frame_index || '-'}</td>
            <td>${v.confidence ? (parseFloat(v.confidence) * 100).toFixed(1) + '%' : '-'}</td>
            <td>
                <a class="evidence-link" onclick="showEvidence('${v.id}', 'vehicle')">Xe</a>
                <a class="evidence-link" onclick="showEvidence('${v.id}', 'plate')"> | Biển số</a>
                <a class="evidence-link" onclick="showEvidence('${v.id}', 'frame')"> | Frame</a>
            </td>
        </tr>
    `).join('');
}

function showEvidence(violationId, type) {
    const modal = document.getElementById('evidenceModal');
    const img = document.getElementById('modalImage');
    img.src = `/api/violations/${violationId}/evidence/${type}`;
    modal.style.display = 'flex';
}

function closeModal() {
    document.getElementById('evidenceModal').style.display = 'none';
}

function toggleRoi() {}

function logout() {
    localStorage.removeItem('token');
    window.location.href = '/';
}

setInterval(() => {
    if (currentVideoId && document.getElementById('btnStop').style.display !== 'none') {
        refreshViolations();
    }
}, 5000);
