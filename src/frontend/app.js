/**
 * ══════════════════════════════════════════════════════════
 *  AI PHẠT NGUỘI GIAO THÔNG — Frontend JavaScript (SPA)
 *  Xử lý: Routing, Upload, SSE stream, Table, Modal,
 *          Toast Notifications, Chart.js
 * ══════════════════════════════════════════════════════════
 */

// ── STATE ───────────────────────────────────────────────
let currentPage = "dashboard";
let currentMode = "camera";
let selectedFiles = [];
let eventSource = null;       // SSE connection (upload processing)
let cameraSource = null;      // SSE connection (camera stream)
let cameraActive = false;
let allViolations = [];       // Cache vi phạm cho filter
let hourlyChart = null;       // Chart.js instance
let currentTaskId = null;     // Tracking process task


// ══════════════════════════════════════════════════════════
// 1. SPA ROUTER
// ══════════════════════════════════════════════════════════

function navigate(page) {
    currentPage = page;

    // Toggle page visibility
    document.getElementById("page-dashboard").classList.toggle("hidden", page !== "dashboard");
    document.getElementById("page-history").classList.toggle("hidden", page !== "history");

    // Toggle nav active state
    const navDash = document.getElementById("nav-dashboard");
    const navHist = document.getElementById("nav-history");

    if (page === "dashboard") {
        navDash.className = "px-5 py-2 rounded-xl text-sm font-semibold transition-all duration-200 nav-active";
        navHist.className = "px-5 py-2 rounded-xl text-sm font-semibold transition-all duration-200 text-slate-400 hover:text-white hover:bg-white/5";
    } else {
        navHist.className = "px-5 py-2 rounded-xl text-sm font-semibold transition-all duration-200 nav-active";
        navDash.className = "px-5 py-2 rounded-xl text-sm font-semibold transition-all duration-200 text-slate-400 hover:text-white hover:bg-white/5";
        loadHistory();
    }

    window.location.hash = page;
}


// ══════════════════════════════════════════════════════════
// 2. MODE SWITCHER (Camera / Upload)
// ══════════════════════════════════════════════════════════

function switchMode(mode) {
    currentMode = mode;

    const btnCam = document.getElementById("mode-camera");
    const btnUp = document.getElementById("mode-upload");
    const panelCam = document.getElementById("panel-camera");
    const panelUp = document.getElementById("panel-upload");

    if (mode === "camera") {
        btnCam.className = "px-6 py-3 text-sm font-semibold transition-all duration-200 bg-indigo-600 text-white";
        btnUp.className = "px-6 py-3 text-sm font-semibold transition-all duration-200 text-slate-400 bg-white/5 hover:bg-white/10";
        panelCam.classList.remove("hidden");
        panelUp.classList.add("hidden");
    } else {
        btnUp.className = "px-6 py-3 text-sm font-semibold transition-all duration-200 bg-indigo-600 text-white";
        btnCam.className = "px-6 py-3 text-sm font-semibold transition-all duration-200 text-slate-400 bg-white/5 hover:bg-white/10";
        panelUp.classList.remove("hidden");
        panelCam.classList.add("hidden");
        // Tắt camera nếu đang chạy
        if (cameraActive) toggleCamera();
    }
}


// ══════════════════════════════════════════════════════════
// 3. TOAST NOTIFICATION
// ══════════════════════════════════════════════════════════

function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    const colors = {
        success: "from-emerald-600 to-emerald-700 border-emerald-500",
        error:   "from-red-600 to-red-700 border-red-500",
        warning: "from-amber-600 to-amber-700 border-amber-500",
        info:    "from-indigo-600 to-indigo-700 border-indigo-500",
    };
    const icons = { success: "✅", error: "❌", warning: "⚠️", info: "ℹ️" };

    const el = document.createElement("div");
    el.className = `toast-enter bg-gradient-to-r ${colors[type]} border rounded-xl px-4 py-3 text-sm font-medium shadow-2xl flex items-center gap-2`;
    el.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
    container.appendChild(el);

    // Auto-dismiss sau 5 giây
    setTimeout(() => {
        el.classList.remove("toast-enter");
        el.classList.add("toast-exit");
        setTimeout(() => el.remove(), 400);
    }, 5000);
}


// ══════════════════════════════════════════════════════════
// 4. FILE UPLOAD (Drag & Drop + Click)
// ══════════════════════════════════════════════════════════

function handleFiles(files) {
    selectedFiles = Array.from(files);
    renderFileList();
    document.getElementById("btn-process").disabled = selectedFiles.length === 0;
}

function renderFileList() {
    const container = document.getElementById("file-list");
    if (selectedFiles.length === 0) {
        container.classList.add("hidden");
        return;
    }
    container.classList.remove("hidden");
    container.innerHTML = selectedFiles.map((f, i) => {
        const isVideo = /\.(mp4|avi|mov)$/i.test(f.name);
        const icon = isVideo ? "🎬" : "🖼️";
        const size = (f.size / 1024 / 1024).toFixed(1);
        return `<div class="glass-card rounded-xl px-4 py-2 flex items-center justify-between">
            <span class="text-sm">${icon} ${f.name} <span class="text-slate-400">(${size}MB)</span></span>
            <button onclick="removeFile(${i})" class="text-red-400 hover:text-red-300 text-xs">✕</button>
        </div>`;
    }).join("");
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderFileList();
    document.getElementById("btn-process").disabled = selectedFiles.length === 0;
}

// Drag & Drop events
document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    if (!dropZone) return;

    ["dragenter", "dragover"].forEach(evt => {
        dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add("drop-active"); });
    });
    ["dragleave", "drop"].forEach(evt => {
        dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove("drop-active"); });
    });
    dropZone.addEventListener("drop", (e) => {
        handleFiles(e.dataTransfer.files);
    });

    // Init page from hash
    const hash = window.location.hash.replace("#", "");
    if (hash === "history") navigate("history");
});


// ══════════════════════════════════════════════════════════
// 5. PROCESSING (Upload → SSE Stream)
// ══════════════════════════════════════════════════════════

async function startProcessing() {
    if (selectedFiles.length === 0) return;

    // 1. Upload files
    const formData = new FormData();
    selectedFiles.forEach(f => formData.append("files", f));

    updateStatus("🔄 Đang tải file…");
    document.getElementById("btn-process").disabled = true;
    showProgress(true);

    try {
        const res = await fetch("/api/upload", { method: "POST", body: formData });
        const data = await res.json();
        if (data.error) {
            showToast(data.error, "error");
            return;
        }

        showToast(`Đã tải ${data.count} file. Bắt đầu xử lý…`, "info");
        updateStatus("🔄 Đang xử lý…");

        currentTaskId = data.task_id;
        document.getElementById("btn-cancel").classList.remove("hidden");
        document.getElementById("btn-cancel").disabled = false;

        // 2. Mở SSE stream để nhận kết quả
        connectProcessSSE(data.task_id);

    } catch (err) {
        showToast("Lỗi upload: " + err.message, "error");
        updateStatus("❌ Lỗi");
        showProgress(false);
    }
}

function connectProcessSSE(taskId) {
    if (eventSource) eventSource.close();

    eventSource = new EventSource(`/api/process/${taskId}`);

    eventSource.onmessage = (e) => {
        const d = JSON.parse(e.data);

        switch (d.type) {
            case "frame":
                showResultImage(d.image);
                updateMetrics(d.stats.vehicles, d.stats.violations);
                if (d.progress) {
                    const pct = Math.round((d.progress.current / d.progress.total) * 100);
                    setProgress(pct, `${d.progress.current}/${d.progress.total}`);
                }
                break;

            case "stage":
                highlightStage(d.stage);
                updateStatus(d.text);
                break;

            case "video_info":
                showToast(`Video: ${d.total_frames} frames, ${d.fps}fps`, "info");
                break;

            case "violation_alert":
                showToast(`🚨 Phát hiện ${d.count} vi phạm mới!`, "warning");
                break;

            case "done":
                updateMetrics(d.total_vehicles, d.total_violations);
                updateStatus("✅ Hoàn tất");
                setProgress(100, "Hoàn tất!");
                showToast(`Xử lý xong! ${d.total_violations} vi phạm.`, "success");
                eventSource.close();
                eventSource = null;
                document.getElementById("btn-process").disabled = false;
                document.getElementById("btn-cancel").classList.add("hidden");
                currentTaskId = null;
                break;

            case "error":
                showToast(d.message, "error");
                updateStatus("❌ Lỗi/Dừng");
                eventSource.close();
                eventSource = null;
                document.getElementById("btn-process").disabled = false;
                document.getElementById("btn-cancel").classList.add("hidden");
                currentTaskId = null;
                break;
        }
    };

    eventSource.onerror = () => {
        eventSource.close();
        eventSource = null;
        document.getElementById("btn-process").disabled = false;
        document.getElementById("btn-cancel").classList.add("hidden");
        currentTaskId = null;
    };
}

async function cancelProcessing() {
    if (!currentTaskId) return;
    try {
        await fetch(`/api/process/cancel/${currentTaskId}`, { method: "POST" });
        showToast("Đang dừng xử lý...", "warning");
        document.getElementById("btn-cancel").disabled = true;
    } catch (err) {
        showToast("Lỗi khi dừng", "error");
    }
}

// ══════════════════════════════════════════════════════════
// 6. CAMERA SIMULATION (SSE Stream)
// ══════════════════════════════════════════════════════════

function toggleCamera() {
    if (cameraActive) {
        // Tắt camera
        if (cameraSource) { cameraSource.close(); cameraSource = null; }
        cameraActive = false;
        document.getElementById("btn-camera").innerHTML = "🔌 Bật Camera";
        document.getElementById("btn-camera").className = "px-5 py-2 rounded-xl text-sm font-bold bg-gradient-to-r from-indigo-500 to-violet-500 hover:from-indigo-400 hover:to-violet-400 transition-all shadow-lg shadow-indigo-500/30";
        document.getElementById("rec-dot").classList.add("hidden");
        document.getElementById("camera-label").textContent = "Camera đã tắt";
        document.getElementById("camera-feed").classList.add("hidden");
        document.getElementById("camera-placeholder").classList.remove("hidden");
        updateStatus("⏹ Camera tắt");
        showToast("Camera đã tắt", "info");
    } else {
        // Bật camera
        cameraActive = true;
        document.getElementById("btn-camera").innerHTML = "⏹ Tắt Camera";
        document.getElementById("btn-camera").className = "px-5 py-2 rounded-xl text-sm font-bold bg-gradient-to-r from-red-500 to-red-600 hover:from-red-400 hover:to-red-500 transition-all shadow-lg shadow-red-500/30";
        document.getElementById("rec-dot").classList.remove("hidden");
        document.getElementById("camera-label").textContent = "Đang kết nối…";
        updateStatus("📡 Đang kết nối camera…");

        cameraSource = new EventSource("/api/camera/stream");

        cameraSource.onmessage = (e) => {
            const d = JSON.parse(e.data);

            switch (d.type) {
                case "camera_start":
                    document.getElementById("camera-label").textContent = `🔴 REC — ${d.source}`;
                    document.getElementById("camera-feed").classList.remove("hidden");
                    document.getElementById("camera-placeholder").classList.add("hidden");
                    updateStatus("🔴 Camera đang giám sát");
                    showToast("Camera trực tiếp đã bật", "success");
                    break;

                case "frame":
                    document.getElementById("camera-feed").src = "data:image/jpeg;base64," + d.image;
                    updateMetrics(d.stats.vehicles, d.stats.violations);
                    break;

                case "violation_alert":
                    showToast(`🚨 Phát hiện ${d.count} vi phạm mới!`, "warning");
                    break;

                case "error":
                    showToast(d.message, "error");
                    toggleCamera();
                    break;
            }
        };

        cameraSource.onerror = () => {
            if (cameraActive) {
                showToast("Mất kết nối camera", "error");
                toggleCamera();
            }
        };
    }
}


// ══════════════════════════════════════════════════════════
// 7. UI HELPERS
// ══════════════════════════════════════════════════════════

function updateMetrics(vehicles, violations) {
    document.getElementById("metric-vehicles").textContent = vehicles;
    document.getElementById("metric-violations").textContent = violations;
}

function updateStatus(text) {
    document.getElementById("metric-status").textContent = text;
}

function showResultImage(b64) {
    const img = document.getElementById("result-display");
    img.src = "data:image/jpeg;base64," + b64;
    img.classList.remove("hidden");
    document.getElementById("result-placeholder").classList.add("hidden");
}

function showProgress(visible) {
    document.getElementById("progress-section").classList.toggle("hidden", !visible);
    if (visible) setProgress(0, "Đang chuẩn bị…");
}

function setProgress(pct, text) {
    document.getElementById("progress-bar").style.width = pct + "%";
    document.getElementById("progress-text").textContent = text;
}

function highlightStage(num) {
    [1, 2, 3].forEach(n => {
        const el = document.getElementById(`stage-${n}`);
        if (!el) return;
        if (n <= num) {
            el.classList.remove("text-slate-500");
            el.classList.add("text-indigo-400");
            el.querySelector("span").className = "w-6 h-6 rounded-full bg-indigo-600 border border-indigo-400 flex items-center justify-center text-[10px] text-white";
        } else {
            el.classList.remove("text-indigo-400");
            el.classList.add("text-slate-500");
            el.querySelector("span").className = "w-6 h-6 rounded-full border border-slate-600 flex items-center justify-center text-[10px]";
        }
    });
}


// ══════════════════════════════════════════════════════════
// 8. HISTORY PAGE — TABLE + CHART + MODAL
// ══════════════════════════════════════════════════════════

async function loadHistory() {
    try {
        // Load violations
        const res = await fetch("/api/violations");
        allViolations = await res.json();
        renderTable(allViolations);

        // Load stats & render chart
        const statsRes = await fetch("/api/violations/stats");
        const stats = await statsRes.json();
        renderChart(stats);
    } catch (err) {
        showToast("Lỗi tải dữ liệu: " + err.message, "error");
    }
}

function renderTable(data) {
    const tbody = document.getElementById("violations-tbody");
    const empty = document.getElementById("violations-empty");

    if (!data || data.length === 0) {
        tbody.innerHTML = "";
        empty.classList.remove("hidden");
        return;
    }
    empty.classList.add("hidden");

    tbody.innerHTML = data.map((v, i) => {
        const evidence = v["Tên file Bằng chứng"] || "";
        return `<tr class="border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors"
                    onclick='openModal(${JSON.stringify(v).replace(/'/g, "&#39;")})'>
            <td class="px-4 py-3 text-slate-400">${i + 1}</td>
            <td class="px-4 py-3">${v["Thời gian vi phạm"] || "—"}</td>
            <td class="px-4 py-3 font-bold text-amber-400">${v["Biển số xe"] || "—"}</td>
            <td class="px-4 py-3 text-xs text-slate-400 truncate max-w-[200px]">${evidence}</td>
            <td class="px-4 py-3 text-red-400">${v["Lỗi"] || "—"}</td>
        </tr>`;
    }).join("");
}

function renderChart(stats) {
    const ctx = document.getElementById("chart-hourly");
    if (!ctx) return;

    // Tạo labels 24 giờ
    const labels = [];
    const values = [];
    for (let h = 0; h < 24; h++) {
        const key = `${String(h).padStart(2, "0")}:00`;
        labels.push(key);
        values.push(stats.hourly[key] || 0);
    }

    // Destroy chart cũ nếu có
    if (hourlyChart) hourlyChart.destroy();

    hourlyChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Số ca vi phạm",
                data: values,
                backgroundColor: "rgba(99,102,241,0.6)",
                borderColor: "rgba(129,140,248,1)",
                borderWidth: 1,
                borderRadius: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: { color: "#94a3b8", font: { size: 10 } },
                    grid: { color: "rgba(255,255,255,0.04)" },
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: "#94a3b8", stepSize: 1 },
                    grid: { color: "rgba(255,255,255,0.04)" },
                },
            },
        },
    });
}

function filterViolations() {
    const from = document.getElementById("filter-date-from").value;
    const to = document.getElementById("filter-date-to").value;

    let filtered = allViolations;

    if (from) {
        filtered = filtered.filter(v => {
            const d = (v["Thời gian vi phạm"] || "").split(" ")[0];
            return d >= from;
        });
    }
    if (to) {
        filtered = filtered.filter(v => {
            const d = (v["Thời gian vi phạm"] || "").split(" ")[0];
            return d <= to;
        });
    }

    renderTable(filtered);
}

async function downloadCSV() {
    try {
        const from = document.getElementById("filter-date-from").value;
        const to = document.getElementById("filter-date-to").value;
        let query = "";
        if (from || to) query = `?from=${from}&to=${to}`;

        const res = await fetch(`/api/violations/download${query}`);
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            showToast(errData.error || "Không có dữ liệu vi phạm", "warning");
            return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "HoSo_PhatNguoi.zip";
        a.click();
        URL.revokeObjectURL(url);
        showToast("Đã tải file ZIP hồ sơ vi phạm", "success");
    } catch (err) {
        showToast("Lỗi tải file", "error");
    }
}

function confirmClearHistory() {
    if (!confirm("⚠️ Bạn có chắc muốn XÓA TOÀN BỘ lịch sử vi phạm?\nHành động này không thể hoàn tác!")) return;
    clearHistory();
}

async function clearHistory() {
    try {
        await fetch("/api/violations", { method: "DELETE" });
        allViolations = [];
        renderTable([]);
        if (hourlyChart) hourlyChart.destroy();
        showToast("Đã xóa toàn bộ lịch sử", "success");
        updateMetrics(0, 0);
    } catch (err) {
        showToast("Lỗi xóa dữ liệu", "error");
    }
}


// ══════════════════════════════════════════════════════════
// 9. MODAL — Xem chi tiết vi phạm
// ══════════════════════════════════════════════════════════

function openModal(violation) {
    const modal = document.getElementById("modal");
    const img = document.getElementById("modal-image");
    const info = document.getElementById("modal-info");

    const evidence = violation["Tên file Bằng chứng"] || "";
    img.src = evidence ? `/api/violations/image/${evidence}` : "";

    info.innerHTML = `
        <div class="glass-card rounded-xl p-4 space-y-2">
            <p><span class="text-slate-400">📅 Thời gian:</span> <span class="font-semibold">${violation["Thời gian vi phạm"] || "—"}</span></p>
            <p><span class="text-slate-400">🏍️ Biển số:</span> <span class="font-bold text-amber-400 text-lg">${violation["Biển số xe"] || "—"}</span></p>
            <p><span class="text-slate-400">🚨 Lỗi:</span> <span class="font-semibold text-red-400">${violation["Lỗi"] || "—"}</span></p>
            <p><span class="text-slate-400">📁 File:</span> <span class="text-xs text-slate-300">${evidence}</span></p>
        </div>
    `;

    modal.classList.remove("hidden");
    modal.classList.add("flex");
}

function closeModal() {
    const modal = document.getElementById("modal");
    modal.classList.add("hidden");
    modal.classList.remove("flex");
}

// Close modal khi click bên ngoài
document.addEventListener("click", (e) => {
    const modal = document.getElementById("modal");
    if (e.target === modal) closeModal();
});

// Close modal bằng phím Escape
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
});
