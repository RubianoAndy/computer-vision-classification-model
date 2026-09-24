// Dirección del modelo publicado en la nube de Teachable Machine.
// Se obtiene en: Exportar modelo → TensorFlow.js → Subir (enlace para compartir).
// El endpoint sirve tres archivos: model.json, metadata.json y weights.bin.
const MODEL_URL = "https://teachablemachine.withgoogle.com/models/UerwbSsVX/";

// Nombre visible, color de la caneca y explicación de cada clase del modelo.
// Código de colores de la Resolución 2184 de 2019 (Colombia).
const CLASSES = {
    "Orgánico": {
        bin: "Caneca verde",
        color: "#2e8b57",
        text: "#ffffff",
        hint: "Restos de comida, cáscaras, residuos de jardín y otros residuos biodegradables.",
    },
    "Reciclable": {
        bin: "Caneca blanca",
        color: "#f4f4f2",
        text: "#1a1a1a",
        hint: "Plástico, vidrio, metales, papel y cartón limpios y secos.",
    },
};
