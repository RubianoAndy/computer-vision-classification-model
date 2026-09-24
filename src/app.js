/* Clasificador de residuos · aplicación web
 * Carga el modelo publicado en la nube de Teachable Machine y lo ejecuta en el
 * navegador con TensorFlow.js. Tres modos: una imagen, conjunto de imágenes
 * (con métricas si las carpetas traen la etiqueta) y cámara en vivo.
 * La decisión final se toma con un umbral configurable: el argmax por defecto
 * (0,5) o el umbral ajustado que corrige el sesgo hacia la clase orgánica.
 */

let model = null;
let labels = [];
let webcam = null;
let cameraLoop = null;
let batchRows = [];
let useAdjustedThreshold = false;

const $ = (id) => document.getElementById(id);
const pct = (p) => `${(p * 100).toFixed(1).replace(".", ",")} %`;
const num = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d).replace(".", ",") : "—");
const normalize = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().trim();

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

/* ─── Umbral de decisión ───────────────────────────────────────────────── */
function currentThreshold() {
    return useAdjustedThreshold ? THRESHOLD.value : THRESHOLD.defaultValue;
}

// Convierte las probabilidades en una etiqueta final según el umbral activo
function decide(probs) {
    const p = probs[THRESHOLD.positive];
    if (p === undefined) {
        return Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
    }
    return p >= currentThreshold() ? THRESHOLD.positive : THRESHOLD.fallback;
}

function setThreshold(adjusted) {
    useAdjustedThreshold = adjusted;
    document.querySelectorAll(".threshold-option").forEach((b) => {
        b.classList.toggle("active", (b.dataset.threshold === "adjusted") === adjusted);
    });
    $("threshold-note").textContent = adjusted
        ? `Umbral ajustado: solo se declara ${THRESHOLD.positive} si su probabilidad es al menos ${pct(THRESHOLD.value)}.`
        : `Umbral por defecto: gana la clase con mayor probabilidad (${pct(THRESHOLD.defaultValue)}).`;
    if (lastSingle) renderResult($("result-single"), lastSingle);
    if (batchRows.length) applyDecisionToBatch();
}

document.querySelectorAll(".threshold-option").forEach((b) => {
    b.addEventListener("click", () => setThreshold(b.dataset.threshold === "adjusted"));
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
        text.textContent = `Modelo en la nube listo · ${labels.join(" / ")}`;
    } catch (err) {
        console.error(err);
        status.classList.remove("loading");
        status.classList.add("error");
        text.textContent = "No se pudo cargar el modelo. Revisa MODEL_URL en config.js";
    }
}

// Devuelve las probabilidades por clase (ordenadas) y la decisión final
async function predict(imageElement) {
    const predictions = await model.predict(imageElement);
    predictions.sort((a, b) => b.probability - a.probability);
    const probs = Object.fromEntries(predictions.map((p) => [p.className, p.probability]));
    return { predictions, probs, decision: decide(probs) };
}

/* ─── Tarjeta de resultado ─────────────────────────────────────────────── */
let lastSingle = null;

function renderResult(container, result) {
    const decision = decide(result.probs);
    const meta = CLASSES[decision] || { bin: "Sin caneca asignada", color: "#ddd", text: "#111", hint: "" };
    const argmax = result.predictions[0].className;
    const bars = result.predictions.map((p) => {
        const m = CLASSES[p.className] || { color: "#999" };
        return `
            <div class="bar-row">
                <span>${p.className}</span>
                <div class="bar-track"><div class="bar-fill" style="width:${(p.probability * 100).toFixed(1)}%;background:${m.color};border:1px solid rgba(0,0,0,.12)"></div></div>
                <span class="bar-value">${pct(p.probability)}</span>
            </div>`;
    }).join("");
    const note = useAdjustedThreshold && decision !== argmax
        ? `<p class="result-note">Con el umbral por defecto la respuesta habría sido <strong>${argmax}</strong>; el umbral ajustado la corrige a <strong>${decision}</strong>.</p>`
        : "";
    container.innerHTML = `
        <div class="result-head">
            <span class="bin-chip" style="background:${meta.color};color:${meta.text}">${meta.bin}</span>
            <div>
                <p class="result-label">${decision}</p>
                <span class="result-conf">Probabilidad de ${decision}: ${pct(result.probs[decision] ?? 0)} · umbral ${useAdjustedThreshold ? "ajustado" : "por defecto"}</span>
            </div>
        </div>
        <p class="result-hint">${meta.hint}</p>
        ${bars}
        ${note}`;
}

/* ─── Modo: una imagen ─────────────────────────────────────────────────── */
const dropzone = $("dropzone-single");
const preview = $("preview-single");

async function classifySingle(file) {
    if (!model || !file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
    $("dropzone-single-inner").hidden = true;
    dropzone.classList.add("has-image");
    await new Promise((resolve) => (preview.onload = resolve));
    lastSingle = await predict(preview);
    renderResult($("result-single"), lastSingle);
    URL.revokeObjectURL(url);
}

$("file-single").addEventListener("change", (e) => classifySingle(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
dropzone.addEventListener("drop", (e) => classifySingle(e.dataTransfer.files[0]));

/* ─── Modo: conjunto de imágenes ───────────────────────────────────────── */
function labelFromPath(file) {
    const parts = (file.webkitRelativePath || "").split("/");
    if (parts.length < 2) return null;
    const folder = normalize(parts[parts.length - 2]);
    return labels.find((l) => normalize(l) === folder) || null;
}

function rowHtml(row, index) {
    const tag = row.correct === null ? `<span class="tag na">sin etiqueta</span>`
        : row.correct ? `<span class="tag ok">acierto</span>` : `<span class="tag bad">error</span>`;
    const meta = CLASSES[row.pred] || { color: "#ddd", text: "#111" };
    return `
        <tr>
            <td>${index + 1}</td>
            <td><img src="${row.url}" alt=""></td>
            <td>${row.file}</td>
            <td>${row.truth || "—"}</td>
            <td><span class="tag" style="background:${meta.color};color:${meta.text};border:1px solid rgba(0,0,0,.12)">${row.pred}</span></td>
            <td>${pct(row.probs[row.pred] ?? 0)}</td>
            <td>${tag}</td>
        </tr>`;
}

// Recalcula la etiqueta de cada fila con el umbral activo, sin volver a
// ejecutar el modelo, y vuelve a dibujar la tabla y las métricas
function applyDecisionToBatch() {
    batchRows.forEach((row) => {
        row.pred = decide(row.probs);
        row.correct = row.truth ? row.truth === row.pred : null;
    });
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
        const truth = labelFromPath(file);
        const row = {
            file: file.name, url, truth, probs: result.probs,
            pred: result.decision,
            correct: truth ? truth === result.decision : null,
        };
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

    const n = labels.length;
    const idx = Object.fromEntries(labels.map((l, i) => [l, i]));
    const cm = Array.from({ length: n }, () => Array(n).fill(0));
    labeled.forEach((r) => cm[idx[r.truth]][idx[r.pred]]++);

    const perClass = labels.map((label, i) => {
        const tp = cm[i][i];
        const fp = cm.reduce((s, row, k) => s + (k !== i ? row[i] : 0), 0);
        const fn = cm[i].reduce((s, v, k) => s + (k !== i ? v : 0), 0);
        const precision = tp + fp ? tp / (tp + fp) : NaN;
        const recall = tp + fn ? tp / (tp + fn) : NaN;
        const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : NaN;
        return { label, precision, recall, f1, support: cm[i].reduce((a, b) => a + b, 0) };
    });
    const accuracy = labeled.filter((r) => r.correct).length / labeled.length;
    const macroF1 = perClass.reduce((s, c) => s + (c.f1 || 0), 0) / n;

    box.innerHTML = `
        <div class="metric-card">
            <h3>Exactitud · umbral ${useAdjustedThreshold ? "ajustado" : "por defecto"}</h3>
            <div class="kpi">${pct(accuracy)}</div>
            <div class="kpi-sub">${labeled.filter((r) => r.correct).length} aciertos de ${labeled.length} imágenes etiquetadas · F1 macro ${num(macroF1)}</div>
        </div>
        <div class="metric-card">
            <h3>Métricas por clase</h3>
            <table>
                <tr><th>Clase</th><th>Precisión</th><th>Exhaust.</th><th>F1</th><th>N</th></tr>
                ${perClass.map((c) => `<tr><td>${c.label}</td><td>${num(c.precision)}</td><td>${num(c.recall)}</td><td>${num(c.f1)}</td><td>${c.support}</td></tr>`).join("")}
            </table>
        </div>
        <div class="metric-card">
            <h3>Matriz de confusión</h3>
            <table class="cm">
                <tr><th>Real \\ Predicha</th>${labels.map((l) => `<th class="rot">${l}</th>`).join("")}</tr>
                ${labels.map((l, i) => `<tr><td>${l}</td>${cm[i].map((v, j) => `<td class="${i === j ? "diag" : ""}">${v}</td>`).join("")}</tr>`).join("")}
            </table>
        </div>`;
    box.hidden = false;
}

function exportCsv() {
    const header = ["file", "true", "pred", "threshold", ...labels].join(",");
    const lines = batchRows.map((r) => [r.file, r.truth || "", r.pred, currentThreshold(), ...labels.map((l) => (r.probs[l] ?? 0).toFixed(6))].join(","));
    const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "predicciones.csv";
    a.click();
    URL.revokeObjectURL(a.href);
}

$("file-batch-folder").addEventListener("change", (e) => classifyBatch(e.target.files));
$("file-batch-files").addEventListener("change", (e) => classifyBatch(e.target.files));
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
        renderResult($("result-camera"), await predict(webcam.canvas));
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

setThreshold(false);
loadModel();
