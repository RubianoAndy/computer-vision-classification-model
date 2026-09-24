// Dirección del modelo publicado en la nube de Teachable Machine.
// Se obtiene en: Exportar modelo → TensorFlow.js → Subir (enlace para compartir).
// El endpoint sirve tres archivos: model.json, metadata.json y weights.bin.
const MODEL_URL = "https://teachablemachine.withgoogle.com/models/UerwbSsVX/";

// Umbral de decisión ajustado. El modelo tiende a llamar "Orgánico" a
// reciclables blandos y de colores vivos, así que en el modo ajustado solo se
// declara Orgánico cuando la probabilidad supera este valor; de lo contrario
// la respuesta es Reciclable. El valor se eligió sobre un segmento de
// validación independiente (ver utils/eval/threshold-selection.py).
const THRESHOLD = {
    positive: "Orgánico",   // clase a la que se le exige el umbral
    fallback: "Reciclable", // clase que se asigna si no lo supera
    value: 0.977,           // umbral ajustado
    defaultValue: 0.5,      // umbral por defecto (equivale al argmax)
};

// Nombres de carpeta que se aceptan como etiqueta de cada clase en el modo
// "Conjunto de imágenes" (se comparan sin tildes ni mayúsculas). Incluye las
// carpetas O y R del dataset original de Kaggle.
const LABEL_ALIASES = {
    "Orgánico": ["organico", "organicos", "organic", "o"],
    "Reciclable": ["reciclable", "reciclables", "recyclable", "r"],
};

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
