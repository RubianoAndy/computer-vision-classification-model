// Uso: node model-inference.js <URL del modelo | carpeta local> <carpeta de prueba> <salida.csv>
// Ejecuta el modelo exportado de Teachable Machine sobre cada imagen del
// segmento de prueba y guarda la clase real y las probabilidades en un CSV.
const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs");
const jpeg = require("jpeg-js");

const [source, testDir, outCsv] = process.argv.slice(2);
const isUrl = /^https?:\/\//.test(source);
const folderOf = (label) => label.normalize("NFD").replace(/[̀-ͯ]/g, "");

// Desde el endpoint en la nube se usa fetch; desde disco, un cargador propio.
const loader = isUrl ? `${source}model.json` : {
    load: async () => {
        const spec = JSON.parse(fs.readFileSync(path.join(source, "model.json")));
        const weights = fs.readFileSync(path.join(source, "weights.bin"));
        return {
            modelTopology: spec.modelTopology,
            weightSpecs: spec.weightsManifest[0].weights,
            weightData: weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.length),
        };
    },
};

// Mismo preprocesamiento de Teachable Machine: recorte cuadrado central,
// redimensión a 224 x 224 y normalización al rango [-1, 1].
function preprocess(file, size) {
    const { width, height, data } = jpeg.decode(fs.readFileSync(file), { useTArray: true, formatAsRGBA: false });
    return tf.tidy(() => {
        const img = tf.tensor3d(data, [height, width, 3], "float32");
        const side = Math.min(width, height);
        const top = Math.floor((height - side) / 2);
        const left = Math.floor((width - side) / 2);
        const crop = img.slice([top, left, 0], [side, side, 3]);
        return tf.image.resizeBilinear(crop, [size, size]).div(127.5).sub(1).expandDims(0);
    });
}

(async () => {
    const metadata = isUrl
        ? await (await fetch(`${source}metadata.json`)).json()
        : JSON.parse(fs.readFileSync(path.join(source, "metadata.json")));
    const { labels, imageSize } = metadata;
    const model = await tf.loadLayersModel(loader);
    console.log(`Modelo cargado desde ${isUrl ? "la nube" : "disco"} · clases: ${labels.join(", ")}`);

    const rows = [["file", "true", ...labels].join(",")];
    for (const label of labels) {
        const dir = path.join(testDir, folderOf(label));
        for (const name of fs.readdirSync(dir).sort()) {
            const input = preprocess(path.join(dir, name), imageSize);
            const probs = await model.predict(input).data();
            input.dispose();
            rows.push([name, label, ...Array.from(probs, (p) => p.toPrecision(8))].join(","));
        }
        console.log(`${label}: listo`);
    }
    fs.writeFileSync(outCsv, rows.join("\n"));
})();
