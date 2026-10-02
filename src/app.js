/* Identificador de plantas carnívoras · aplicación web
 * Carga el modelo publicado en la nube de Teachable Machine y lo ejecuta en el
 * navegador con TensorFlow.js. Tres modos: una imagen, conjunto de imágenes
 * (con métricas multiclase si las carpetas traen la etiqueta) y cámara en vivo.
 * Si la probabilidad de la clase ganadora no alcanza el umbral de confianza,
 * la respuesta es "no estoy seguro" con las dos opciones más probables.
 */

let model = null;
let labels = [];
let webcam = null;
let cameraLoop = null;
let batchRows = [];
let useConfidence = CONFIDENCE.enabledByDefault;
let lastSingle = null;

const UNSURE = "No estoy seguro";
const $ = (id) => document.getElementById(id);
const pct = (p) => `${(p * 100).toFixed(1).replace(".", ",")} %`;
const num = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d).replace(".", ",") : "—");
const normalize = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().trim();
// Formas en que puede aparecer una clase en carpetas o nombres de archivo:
// "No carnivora", "no_carnivora", "no-carnivora", "nocarnivora"
const slugs = (label) => {
    const n = normalize(label);
    return [...new Set([n, n.replace(/\s+/g, "_"), n.replace(/\s+/g, "-"), n.replace(/\s+/g, "")])];
};

/* ─── Pestañas ─────────────────────────────────────────────────────────── */
function showTab(name) {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
    if (name !== "camera") stopCamera();
}

document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => showTab(tab.dataset.tab));
});

// Enlaces de la barra superior y de la portada que abren una pestaña concreta
document.querySelectorAll("[data-goto]").forEach((link) => {
    link.addEventListener("click", () => showTab(link.dataset.goto));
});

/* ─── Umbral de confianza ──────────────────────────────────────────────── */
// Convierte las probabilidades ordenadas en la respuesta final
function decide(predictions) {
    const [top, second] = predictions;
    const unsure = useConfidence && top.probability < CONFIDENCE.value;
    return { label: unsure ? UNSURE : top.className, top, second, unsure };
}

function setConfidence(enabled) {
    useConfidence = enabled;
    document.querySelectorAll(".threshold-option").forEach((b) => {
        b.classList.toggle("active", (b.dataset.threshold === "on") === enabled);
    });
    $("threshold-note").textContent = enabled
        ? `Si la clase ganadora no alcanza ${pct(CONFIDENCE.value)} de probabilidad, la respuesta es "${UNSURE}" y se muestran las dos opciones más probables.`
        : "Siempre responde la clase con mayor probabilidad, aunque la confianza sea baja.";
    if (lastSingle) renderResult($("result-single"), lastSingle);
    if (batchRows.length) applyDecisionToBatch();
}

document.querySelectorAll(".threshold-option").forEach((b) => {
    b.addEventListener("click", () => setConfidence(b.dataset.threshold === "on"));
});

/* ─── Carga del modelo desde el endpoint de Teachable Machine ─────────── */
async function loadModel() {
    const status = $("model-status");
    const text = $("model-status-text");
    $("about-model-url").textContent = `Endpoint del modelo: ${MODEL_URL}`;
    try {
        model = await tmImage.load(`${MODEL_URL}model.json`, `${MODEL_URL}metadata.json`);
        labels = model.getClassLabels();
        status.classList.remove("loading");
        status.classList.add("ready");
        text.textContent = `Modelo en la nube listo · ${labels.length} clases`;
        fillBatchLabelOptions();
    } catch (err) {
        console.error(err);
        status.classList.remove("loading");
        status.classList.add("error");
        text.textContent = "No se pudo cargar el modelo. Revisa MODEL_URL en config.js";
    }
}

// Recorte cuadrado central de la imagen original (tamaño natural, no el que
// tiene en pantalla), llevado a 224 × 224: lo mismo que hace Teachable Machine
// al entrenar y que el script de evaluación. Sin esto, la biblioteca escala el
// elemento <img> con sus dimensiones en pantalla y la foto llega deformada.
const squareCanvas = document.createElement("canvas");
function toSquare(source) {
    const w = source.naturalWidth || source.videoWidth || source.width;
    const h = source.naturalHeight || source.videoHeight || source.height;
    const side = Math.min(w, h);
    squareCanvas.width = squareCanvas.height = 224;
    squareCanvas.getContext("2d").drawImage(source, (w - side) / 2, (h - side) / 2, side, side, 0, 0, 224, 224);
    return squareCanvas;
}

// Devuelve las probabilidades por clase ordenadas de mayor a menor
async function predict(imageElement) {
    const input = imageElement instanceof HTMLImageElement ? toSquare(imageElement) : imageElement;
    const predictions = await model.predict(input);
    predictions.sort((a, b) => b.probability - a.probability);
    const probs = Object.fromEntries(predictions.map((p) => [p.className, p.probability]));
    return { predictions, probs };
}

/* ─── Tarjeta de resultado ─────────────────────────────────────────────── */
function careHtml(label) {
    const meta = CLASSES[label];
    if (!meta) return "";
    const rows = Object.entries(meta.care).map(([k, v]) => `<div class="care-row"><dt>${k}</dt><dd>${v}</dd></div>`).join("");
    const video = meta.video
        ? `<a class="button ghost small-button" href="${meta.video}" target="_blank" rel="noopener">Ver el video de ${label} en el canal ▶</a>`
        : `<a class="button ghost small-button" href="${CHANNEL_URL}" target="_blank" rel="noopener">Ver más en el canal ▶</a>`;
    return `
        <details class="care" open>
            <summary>Ficha de cuidados · ${label}${meta.common ? ` (${meta.common})` : ""}</summary>
            <dl>${rows}</dl>
            ${video}
        </details>`;
}

function renderResult(container, result, { compact = false } = {}) {
    const d = decide(result.predictions);
    const meta = d.unsure
        ? { color: "#b8c0cc", text: "#1a1a1a", trap: "Confianza baja", common: "" }
        : (CLASSES[d.label] || { color: "#ddd", text: "#111", trap: "", common: "" });
    const bars = result.predictions.slice(0, compact ? 3 : labels.length).map((p) => {
        const m = CLASSES[p.className] || { color: "#999" };
        return `
            <div class="bar-row">
                <span>${p.className}</span>
                <div class="bar-track"><div class="bar-fill" style="width:${(p.probability * 100).toFixed(1)}%;background:${m.color};border:1px solid rgba(0,0,0,.12)"></div></div>
                <span class="bar-value">${pct(p.probability)}</span>
            </div>`;
    }).join("");
    const subtitle = d.unsure
        ? `Lo más probable es <strong>${d.top.className}</strong> (${pct(d.top.probability)}); la segunda opción es <strong>${d.second.className}</strong> (${pct(d.second.probability)}).`
        : `Confianza ${pct(d.top.probability)} · segunda opción: ${d.second.className} (${pct(d.second.probability)})`;
    const hint = d.unsure
        ? "La foto no alcanza el umbral de confianza. Prueba con la trampa más cerca, centrada y con buena luz."
        : (CLASSES[d.label]?.hint || "");
    container.innerHTML = `
        <div class="result-head">
            <span class="bin-chip" style="background:${meta.color};color:${meta.text}">${meta.trap || "—"}</span>
            <div>
                <p class="result-label">${d.label}${meta.common ? `<span class="result-common">${meta.common}</span>` : ""}</p>
                <span class="result-conf">${subtitle}</span>
            </div>
        </div>
        <p class="result-hint">${hint}</p>
        ${bars}
        ${compact || d.unsure ? "" : careHtml(d.label)}`;
}

/* ─── Modo: una imagen ─────────────────────────────────────────────────── */
const dropzone = $("dropzone-single");
const preview = $("preview-single");

async function classifySingle(file) {
    if (!model || !file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
    $("dropzone-note").hidden = false;
    $("dropzone-single-inner").hidden = true;
    dropzone.classList.add("has-image");
    await new Promise((resolve) => (preview.onload = resolve));
    lastSingle = await predict(preview);
    renderResult($("result-single"), lastSingle);
    URL.revokeObjectURL(url);
}

// Fotos de muestra: se descargan como archivo y siguen el mismo camino que una foto del usuario
function renderSamples() {
    const row = $("samples");
    if (!row) return;
    row.innerHTML = SAMPLES.map((s) => `
        <button class="sample" type="button" data-src="${s.src}" title="${s.label}">
            <img src="${s.src}" alt="${s.label}" loading="lazy">
            <span>${s.label}</span>
        </button>`).join("");
    row.querySelectorAll(".sample").forEach((b) => b.addEventListener("click", async () => {
        const blob = await (await fetch(b.dataset.src)).blob();
        classifySingle(new File([blob], b.dataset.src.split("/").pop(), { type: blob.type || "image/jpeg" }));
        $("identificador").scrollIntoView({ behavior: "smooth", block: "start" });
    }));
}

$("file-single").addEventListener("change", (e) => classifySingle(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
dropzone.addEventListener("drop", (e) => classifySingle(e.dataTransfer.files[0]));

/* ─── Modo: conjunto de imágenes ───────────────────────────────────────── */
function fillBatchLabelOptions() {
    const select = $("batch-label");
    labels.forEach((l) => {
        const opt = document.createElement("option");
        opt.value = l;
        opt.textContent = `Todas son ${l}`;
        select.insertBefore(opt, select.lastElementChild);
    });
}

// Etiqueta real a partir de la carpeta que contiene la imagen
function labelFromPath(file) {
    const parts = (file.webkitRelativePath || "").split("/");
    if (parts.length < 2) return null;
    const folder = normalize(parts[parts.length - 2]);
    return labels.find((l) => slugs(l).includes(folder)) || null;
}

// Etiqueta real a partir del prefijo del nombre (dionaea_012.jpg, no_carnivora_003.jpg)
function labelFromName(name) {
    const n = normalize(name);
    const hits = labels.filter((l) => slugs(l).some((s) => n.startsWith(`${s}_`) || n.startsWith(`${s}-`)));
    // "no_carnivora" también empieza por "no"... se elige la coincidencia más larga
    return hits.sort((a, b) => b.length - a.length)[0] || null;
}

// Clase real según el selector: automática (carpeta y luego prefijo), una
// clase fija para todas las imágenes, o ninguna
function truthFor(row) {
    const mode = $("batch-label").value;
    if (mode === "none") return null;
    if (labels.includes(mode)) return mode;
    return labelFromPath(row.fileObj) || labelFromName(row.file);
}

function rowHtml(row, index) {
    const tag = row.correct === null ? `<span class="tag na">${row.truth ? "Sin responder" : "Sin etiqueta"}</span>`
        : row.correct ? `<span class="tag ok">Acierto</span>` : `<span class="tag bad">Error</span>`;
    const meta = row.unsure ? { color: "#b8c0cc", text: "#1a1a1a" } : (CLASSES[row.pred] || { color: "#ddd", text: "#111" });
    return `
        <tr>
            <td>${index + 1}</td>
            <td><img src="${row.url}" alt=""></td>
            <td>${row.file}</td>
            <td>${row.truth || "—"}</td>
            <td><span class="tag" style="background:${meta.color};color:${meta.text};border:1px solid rgba(0,0,0,.12)">${row.pred}</span></td>
            <td>${pct(row.top.probability)}</td>
            <td>${row.second.className} (${pct(row.second.probability)})</td>
            <td>${tag}</td>
        </tr>`;
}

function applyRow(row) {
    const d = decide(row.predictions);
    row.truth = truthFor(row);
    row.pred = d.label;
    row.top = d.top;
    row.second = d.second;
    row.unsure = d.unsure;
    // Las imágenes sin responder no cuentan como acierto ni como error
    row.correct = row.truth && !row.unsure ? row.truth === row.pred : null;
}

// Recalcula cada fila con el umbral activo, sin volver a ejecutar el modelo
function applyDecisionToBatch() {
    batchRows.forEach(applyRow);
    $("batch-table").querySelector("tbody").innerHTML = batchRows.map(rowHtml).join("");
    renderMetrics();
}

async function classifyBatch(fileList) {
    if (!model) return;
    const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    files.sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));

    const tbody = $("batch-table").querySelector("tbody");
    batchRows.forEach((r) => URL.revokeObjectURL(r.url));
    tbody.innerHTML = "";
    document.querySelector(".table-wrap").scrollTop = 0;
    $("batch-table").hidden = false;
    $("metrics").hidden = true;
    $("progress").hidden = false;
    $("export-csv").disabled = true;
    batchRows = [];

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.src = url;
        await new Promise((resolve) => (img.onload = resolve));
        const result = await predict(img);
        const row = { file: file.name, fileObj: file, url, probs: result.probs, predictions: result.predictions };
        applyRow(row);
        batchRows.push(row);
        tbody.insertAdjacentHTML("beforeend", rowHtml(row, i));
        $("progress-bar").style.width = `${((i + 1) / files.length) * 100}%`;
        $("progress-text").textContent = `${i + 1} de ${files.length} imágenes`;
    }

    $("export-csv").disabled = false;
    renderMetrics();
}

function renderMetrics() {
    const labeled = batchRows.filter((r) => r.truth);
    const box = $("metrics");
    if (!labeled.length) { box.hidden = true; return; }

    const answered = labeled.filter((r) => !r.unsure);
    const n = labels.length;
    const idx = Object.fromEntries(labels.map((l, i) => [l, i]));
    const cm = Array.from({ length: n }, () => Array(n).fill(0));
    answered.forEach((r) => cm[idx[r.truth]][idx[r.pred]]++);

    const perClass = labels.map((label, i) => {
        const tp = cm[i][i];
        const fp = cm.reduce((s, row, k) => s + (k !== i ? row[i] : 0), 0);
        const fn = cm[i].reduce((s, v, k) => s + (k !== i ? v : 0), 0);
        const precision = tp + fp ? tp / (tp + fp) : NaN;
        const recall = tp + fn ? tp / (tp + fn) : NaN;
        const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : NaN;
        return { label, precision, recall, f1, support: labeled.filter((r) => r.truth === label).length };
    });
    const hits = answered.filter((r) => r.correct).length;
    const accuracy = answered.length ? hits / answered.length : NaN;
    const present = perClass.filter((c) => c.support > 0);
    const macroF1 = present.reduce((s, c) => s + (c.f1 || 0), 0) / present.length;
    const unsureNote = useConfidence
        ? `${labeled.length - answered.length} sin responder por confianza baja (cobertura ${pct(answered.length / labeled.length)})`
        : "sin umbral de confianza";

    box.innerHTML = `
        <div class="metric-card">
            <h3>Exactitud en las imágenes respondidas</h3>
            <div class="kpi">${Number.isFinite(accuracy) ? pct(accuracy) : "—"}</div>
            <div class="kpi-sub">${hits} aciertos de ${answered.length} respondidas · ${unsureNote} · F1 macro ${num(macroF1)}</div>
        </div>
        <div class="metric-card">
            <h3>Métricas por clase</h3>
            <table>
                <tr><th>Clase</th><th>Precisión</th><th>Exhaust.</th><th>F1</th><th>N</th></tr>
                ${perClass.map((c) => `<tr><td>${c.label}</td><td>${num(c.precision)}</td><td>${num(c.recall)}</td><td>${num(c.f1)}</td><td>${c.support}</td></tr>`).join("")}
            </table>
        </div>
        <div class="metric-card wide">
            <h3>Matriz de confusión (filas: clase real · columnas: predicha)</h3>
            <table class="cm">
                <tr><th></th>${labels.map((l) => `<th class="rot">${l.replace("No carnivora", "No carn.")}</th>`).join("")}</tr>
                ${labels.map((l, i) => `<tr><td>${l}</td>${cm[i].map((v, j) => `<td class="${i === j ? "diag" : v ? "off" : ""}">${v}</td>`).join("")}</tr>`).join("")}
            </table>
        </div>`;
    box.hidden = false;
}

function exportCsv() {
    const header = ["file", "true", "pred", "confidence_threshold", ...labels].join(",");
    const lines = batchRows.map((r) => [r.file, r.truth || "", r.pred, useConfidence ? CONFIDENCE.value : 0, ...labels.map((l) => (r.probs[l] ?? 0).toFixed(6))].join(","));
    const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "predicciones.csv";
    a.click();
    URL.revokeObjectURL(a.href);
}

$("file-batch-folder").addEventListener("change", (e) => classifyBatch(e.target.files));
$("file-batch-files").addEventListener("change", (e) => classifyBatch(e.target.files));
$("batch-label").addEventListener("change", () => { if (batchRows.length) applyDecisionToBatch(); });
$("export-csv").addEventListener("click", exportCsv);

/* ─── Modo: cámara ─────────────────────────────────────────────────────── */
async function startCamera() {
    if (!model || webcam) return;
    webcam = new tmImage.Webcam(360, 360, true);
    await webcam.setup();
    await webcam.play();
    const container = $("webcam-container");
    container.innerHTML = "";
    container.appendChild(webcam.canvas);
    $("camera-start").disabled = true;
    $("camera-stop").disabled = false;
    const loop = async () => {
        if (!webcam) return;
        webcam.update();
        renderResult($("result-camera"), await predict(webcam.canvas), { compact: true });
        cameraLoop = window.requestAnimationFrame(loop);
    };
    loop();
}

function stopCamera() {
    if (!webcam) return;
    window.cancelAnimationFrame(cameraLoop);
    webcam.stop();
    webcam = null;
    $("webcam-container").innerHTML = `<span class="muted">La cámara está apagada.</span>`;
    $("camera-start").disabled = false;
    $("camera-stop").disabled = true;
}

$("camera-start").addEventListener("click", startCamera);
$("camera-stop").addEventListener("click", stopCamera);

/* ─── Sección de géneros (fichas) ──────────────────────────────────────── */
function renderGenera() {
    const grid = $("genera");
    if (!grid) return;
    grid.innerHTML = Object.entries(CLASSES).filter(([l]) => l !== "No carnivora").map(([label, m]) => `
        <article class="genus" style="--c:${m.color}">
            <span class="genus-trap">${m.trap}</span>
            <h3>${label}</h3>
            <p class="genus-common">${m.common}</p>
            <p class="genus-hint">${m.hint}</p>
            ${careHtml(label).replace(" open>", ">")}
        </article>`).join("");
}

/* ─── Redes sociales ───────────────────────────────────────────────────── */
const ICONS = {
    youtube: `<path d="M23.5 6.2a3 3 0 0 0-2.1-2.1C19.5 3.6 12 3.6 12 3.6s-7.5 0-9.4.5A3 3 0 0 0 .5 6.2 31 31 0 0 0 0 12a31 31 0 0 0 .5 5.8 3 3 0 0 0 2.1 2.1c1.9.5 9.4.5 9.4.5s7.5 0 9.4-.5a3 3 0 0 0 2.1-2.1A31 31 0 0 0 24 12a31 31 0 0 0-.5-5.8zM9.6 15.6V8.4l6.2 3.6-6.2 3.6z"/>`,
    tiktok: `<path d="M16.6 5.8A4.3 4.3 0 0 1 15.5 3h-3.1v12.4a2.6 2.6 0 1 1-1.8-2.5V9.7a5.7 5.7 0 1 0 4.9 5.7V9.1a7.4 7.4 0 0 0 4.3 1.4V7.4a4.3 4.3 0 0 1-3.2-1.6z"/>`,
    instagram: `<path d="M12 2.2c3.2 0 3.6 0 4.8.1 3.3.1 4.8 1.7 4.9 4.9.1 1.3.1 1.6.1 4.8s0 3.6-.1 4.8c-.1 3.2-1.7 4.8-4.9 4.9-1.3.1-1.6.1-4.8.1s-3.6 0-4.8-.1c-3.3-.1-4.8-1.7-4.9-4.9C2.2 15.6 2.2 15.2 2.2 12s0-3.6.1-4.8C2.4 3.9 4 2.4 7.2 2.3c1.2-.1 1.6-.1 4.8-.1zM12 0C8.7 0 8.3 0 7.1.1 2.7.3.3 2.7.1 7.1 0 8.3 0 8.7 0 12s0 3.7.1 4.9c.2 4.4 2.6 6.8 7 7 1.2.1 1.6.1 4.9.1s3.7 0 4.9-.1c4.4-.2 6.8-2.6 7-7 .1-1.2.1-1.6.1-4.9s0-3.7-.1-4.9c-.2-4.4-2.6-6.8-7-7C15.7 0 15.3 0 12 0zm0 5.8a6.2 6.2 0 1 0 0 12.4 6.2 6.2 0 0 0 0-12.4zM12 16a4 4 0 1 1 0-8 4 4 0 0 1 0 8zm6.4-11.8a1.4 1.4 0 1 0 0 2.9 1.4 1.4 0 0 0 0-2.9z"/>`,
    facebook: `<path d="M24 12a12 12 0 1 0-13.9 11.9v-8.4H7.1V12h3V9.4c0-3 1.8-4.7 4.5-4.7 1.3 0 2.7.2 2.7.2v3h-1.5c-1.5 0-2 .9-2 1.9V12h3.3l-.5 3.5h-2.8v8.4A12 12 0 0 0 24 12z"/>`,
    linkedin: `<path d="M20.4 20.5h-3.6v-5.6c0-1.3 0-3-1.8-3s-2.1 1.4-2.1 2.9v5.7H9.4V9h3.4v1.6a3.7 3.7 0 0 1 3.4-1.9c3.6 0 4.3 2.4 4.3 5.5v6.3zM5.3 7.4a2.1 2.1 0 1 1 0-4.1 2.1 2.1 0 0 1 0 4.1zm1.8 13.1H3.6V9h3.5v11.5zM22.2 0H1.8C.8 0 0 .8 0 1.7v20.6c0 .9.8 1.7 1.8 1.7h20.4c1 0 1.8-.8 1.8-1.7V1.7C24 .8 23.2 0 22.2 0z"/>`,
};

function renderSocial() {
    document.querySelectorAll("[data-social]").forEach((box) => {
        const withText = box.dataset.social === "text";
        box.innerHTML = SOCIAL.map((s) => `
            <a class="social" href="${s.url}" target="_blank" rel="noopener" aria-label="${s.name}" title="${s.name}">
                <svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[s.icon]}</svg>${withText ? `<span>${s.name}</span>` : ""}
            </a>`).join("");
    });
}

renderSocial();
renderSamples();
renderGenera();
setConfidence(CONFIDENCE.enabledByDefault);
loadModel();
