// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  file: null,        // File object
  mode: null,        // 'image' | 'video'
  resultPath: null,  // last saved result path
  taskId: null,      // current video task
  sse: null,         // EventSource
};

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const dropMeta       = document.getElementById("drop-meta");
const dropLabel      = document.getElementById("drop-label");
const sharedControls = document.getElementById("shared-controls");
const videoControls  = document.getElementById("video-controls");
const tabRemove      = document.getElementById("tab-remove");
const softControls   = document.getElementById("soft-controls");
const gifFpsRow      = document.getElementById("gif-fps-row");
const btnConvert     = document.getElementById("btn-convert");
const errorBanner    = document.getElementById("error-banner");
const app            = document.getElementById("app");

// ─── File handling ────────────────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});

function setFile(f) {
  const type = f.type;
  if (!type.startsWith("image/") && !type.startsWith("video/")) {
    showError("Поддерживаются только изображения и видео");
    return;
  }
  clearError();
  state.file = f;
  state.mode = type.startsWith("image/") ? "image" : "video";

  dropZone.classList.add("has-file");
  dropLabel.textContent = f.name;
  dropMeta.textContent = (f.size / 1024).toFixed(0) + " KB";

  applyMode(state.mode);
  validateForm();
}

function applyMode(mode) {
  const isVideo = mode === "video";
  app.classList.toggle("video-mode", isVideo);
  dropZone.classList.toggle("video-mode", isVideo);
  videoControls.classList.toggle("hidden", !isVideo);
  tabRemove.classList.toggle("hidden", isVideo);

  // Reset bg-mode to keep if remove was selected and now in video mode
  if (isVideo && getActiveTab("#bg-mode-tabs") === "remove") {
    setActiveTab("#bg-mode-tabs", "keep");
    softControls.classList.add("hidden");
  }
}

// ─── Sliders ──────────────────────────────────────────────────────────────────
document.querySelectorAll("input[type=range]").forEach(input => {
  const valId = "val-" + input.id.replace("ctrl-", "");
  const valEl = document.getElementById(valId);
  if (valEl) {
    input.addEventListener("input", () => { valEl.textContent = input.value; });
  }
});

// ─── Layer toggles ────────────────────────────────────────────────────────────
document.querySelectorAll(".toggle[data-layer]").forEach(btn => {
  btn.addEventListener("click", () => {
    btn.classList.toggle("active");
    validateForm();
  });
});

function getActiveLayers() {
  return [...document.querySelectorAll(".toggle[data-layer].active")]
    .map(b => b.dataset.layer);
}

// ─── BG mode tabs ─────────────────────────────────────────────────────────────
document.querySelectorAll(".tab[data-mode]").forEach(btn => {
  btn.addEventListener("click", () => {
    setActiveTab("#bg-mode-tabs", btn.dataset.mode);
    softControls.classList.toggle("hidden", btn.dataset.mode !== "soft");
    validateForm();
  });
});

function setActiveTab(containerSelector, value) {
  document.querySelectorAll(containerSelector + " .tab").forEach(b => {
    b.classList.toggle("active", b.dataset.mode === value);
  });
}

function getActiveTab(containerSelector) {
  const active = document.querySelector(containerSelector + " .tab.active");
  return active ? active.dataset.mode : null;
}

// ─── Conditional controls ─────────────────────────────────────────────────────
document.getElementById("ctrl-gif").addEventListener("change", function () {
  gifFpsRow.classList.toggle("hidden", !this.checked);
});

// ─── Validation ───────────────────────────────────────────────────────────────
function validateForm() {
  const noFile   = !state.file;
  const noLayers = getActiveLayers().length === 0;
  const softEmpty = getActiveTab("#bg-mode-tabs") === "soft" &&
    !document.getElementById("ctrl-bg-chars").value.trim();

  btnConvert.disabled = noFile || noLayers || softEmpty;
  btnConvert.title = noLayers ? "Выбери хотя бы один слой"
    : softEmpty ? "--bg-chars не может быть пустым"
    : "";
}

document.getElementById("ctrl-bg-chars").addEventListener("input", validateForm);

// ─── Collect params ───────────────────────────────────────────────────────────
function collectParams() {
  return {
    width:      parseInt(document.getElementById("ctrl-width").value),
    font_size:  parseInt(document.getElementById("ctrl-font-size").value),
    jitter:     parseInt(document.getElementById("ctrl-jitter").value),
    threshold:  parseFloat(document.getElementById("ctrl-threshold").value),
    layers:     getActiveLayers(),
    bg_mode:    getActiveTab("#bg-mode-tabs"),
    bg_opacity: parseFloat(document.getElementById("ctrl-bg-opacity").value),
    bg_chars:   document.getElementById("ctrl-bg-chars").value,
    profile:    document.getElementById("ctrl-profile").value,
    // video
    fps:        parseInt(document.getElementById("ctrl-fps").value),
    workers:    parseInt(document.getElementById("ctrl-workers").value),
    frame_step: document.getElementById("ctrl-frame-step").value || null,
    preview:    document.getElementById("ctrl-preview").checked,
    gif:        document.getElementById("ctrl-gif").checked,
    gif_fps:    parseInt(document.getElementById("ctrl-gif-fps").value),
  };
}

// ─── Convert ──────────────────────────────────────────────────────────────────
btnConvert.addEventListener("click", async () => {
  clearError();
  btnConvert.disabled = true;
  btnConvert.textContent = "⏳ Обработка...";

  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("params", JSON.stringify(collectParams()));
  fd.append("mode", state.mode);

  try {
    const resp = await fetch("/api/convert", { method: "POST", body: fd });
    const data = await resp.json();
    if (data.error) { showError(data.error); return; }

    if (state.mode === "image") {
      showImageResult(data.result_url, data.result_path);
    } else {
      startVideoProgress(data.task_id, data.result_path, collectParams().gif);
    }
  } catch (e) {
    showError("Ошибка соединения: " + e.message);
  } finally {
    btnConvert.disabled = false;
    btnConvert.textContent = "▶ Конвертировать";
  }
});

// ─── Image result ─────────────────────────────────────────────────────────────
function showImageResult(url, path) {
  document.getElementById("result-placeholder").classList.add("hidden");
  document.getElementById("video-result").classList.add("hidden");
  const box = document.getElementById("image-result");
  box.classList.remove("hidden");
  document.getElementById("result-img").src = url + "?t=" + Date.now();
  document.getElementById("result-path-label").textContent = path;
  state.resultPath = path;
}

document.getElementById("btn-copy-path").addEventListener("click", () => {
  navigator.clipboard.writeText(state.resultPath);
});

document.getElementById("btn-open-folder").addEventListener("click", () => {
  openFolder(state.resultPath);
});

// ─── Video progress ───────────────────────────────────────────────────────────
function startVideoProgress(taskId, resultPath, gifRequested) {
  state.taskId = taskId;
  state.resultPath = resultPath;

  document.getElementById("result-placeholder").classList.add("hidden");
  document.getElementById("image-result").classList.add("hidden");
  const box = document.getElementById("video-result");
  box.classList.remove("hidden");
  document.getElementById("progress-box").classList.remove("hidden");
  document.getElementById("preview-img").classList.add("hidden");
  document.getElementById("video-actions").classList.add("hidden");

  if (state.sse) state.sse.close();
  state.sse = new EventSource("/api/progress/" + taskId);

  state.sse.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.error) {
      state.sse.close();
      showError(d.error);
      return;
    }
    if (d.done) {
      state.sse.close();
      showVideoResult(d.result_url, resultPath, gifRequested, d.gif_url);
      return;
    }
    const pct = d.total ? Math.round(d.frame / d.total * 100) : 0;
    document.getElementById("progress-bar").style.width = pct + "%";
    document.getElementById("progress-count").textContent = d.frame + " / " + d.total;
    if (d.preview_url) {
      document.getElementById("preview-img").src = d.preview_url;
      document.getElementById("preview-img").classList.remove("hidden");
    }
  };
  state.sse.onerror = () => { state.sse.close(); showError("Ошибка соединения SSE"); };
}

function showVideoResult(mp4Url, resultPath, gifRequested, gifUrl) {
  document.getElementById("progress-box").classList.add("hidden");
  const previewImg = document.getElementById("preview-img");
  if (mp4Url) {
    previewImg.src = mp4Url.replace(".mp4", "_preview.png") + "?t=" + Date.now();
    previewImg.classList.remove("hidden");
  }
  const actions = document.getElementById("video-actions");
  actions.classList.remove("hidden");
  document.getElementById("video-path-label").textContent = resultPath;
  document.getElementById("btn-save-mp4").onclick = () => window.open(mp4Url);
  const btnGif = document.getElementById("btn-save-gif");
  if (gifRequested && gifUrl) {
    btnGif.classList.remove("hidden");
    btnGif.onclick = () => window.open(gifUrl);
  } else {
    btnGif.classList.add("hidden");
  }
}

document.getElementById("btn-open-folder-video").addEventListener("click", () => {
  openFolder(state.resultPath);
});

// ─── Helpers ──────────────────────────────────────────────────────────────────
function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove("hidden");
}
function clearError() { errorBanner.classList.add("hidden"); }

async function openFolder(path) {
  await fetch("/api/open-folder", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
}
