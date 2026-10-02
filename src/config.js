// Dirección del modelo publicado en la nube de Teachable Machine.
// Se obtiene en: Exportar modelo → TensorFlow.js → Subir (enlace para compartir).
// El endpoint sirve tres archivos: model.json, metadata.json y model.weights.bin.
const MODEL_URL = "https://teachablemachine.withgoogle.com/models/RmIb0tr6_/";

// Umbral de confianza. El modelo siempre elige alguna clase; si la probabilidad
// de la clase ganadora no alcanza este valor, la aplicación responde "no estoy
// seguro" y muestra las dos opciones más probables. Con 0,8 el modelo responde
// el 89 % de las imágenes de prueba con una exactitud de 0,91 (ver
// utils/eval/cloud-carnivoras-metrics.json).
const CONFIDENCE = {
    value: 0.8,
    enabledByDefault: true,
};

// Canal de YouTube del autor. Las fichas enlazan a los videos de cada género.
const CHANNEL_URL = "https://www.youtube.com/@AndyRubiano";

// Ficha de cada clase del modelo: nombre, tipo de trampa, color para las
// barras, descripción, cuidados básicos y video del canal (vacío si aún no hay).
const CLASSES = {
    "Dionaea": {
        common: "Venus atrapamoscas",
        trap: "Trampa de cierre rápido",
        color: "#2e8b57",
        text: "#ffffff",
        hint: "Hojas terminadas en dos lóbulos con pelos sensores que se cierran en menos de un segundo cuando un insecto los toca.",
        care: {
            "Luz": "Sol directo, al menos 4 a 6 horas diarias; a la sombra pierde color y las trampas se vuelven débiles.",
            "Agua": "Solo agua destilada, de lluvia o de ósmosis, por bandeja, con 1 a 2 cm de agua permanente en temporada de crecimiento.",
            "Sustrato": "Turba rubia con perlita a partes iguales, sin abono ni tierra para plantas.",
            "Dormancia": "Necesita un reposo invernal de 3 a 4 meses con frío y menos agua; sin él se debilita.",
            "Alimentación": "No hace falta alimentarla. No cierres las trampas por juego: cada una se abre pocas veces y luego muere.",
        },
        video: "",
    },
    "Drosera": {
        common: "Rocío de sol",
        trap: "Trampa adhesiva",
        color: "#c0392b",
        text: "#ffffff",
        hint: "Hojas cubiertas de tentáculos con gotas de mucílago brillante que atrapan y envuelven a la presa. Hay rosetas, especies erectas, pigmeas y tuberosas.",
        care: {
            "Luz": "Mucha luz; con sol directo o luz artificial intensa los tentáculos se tornan rojos y producen más rocío.",
            "Agua": "Agua destilada o de lluvia, por bandeja; la mayoría tolera el sustrato siempre húmedo.",
            "Sustrato": "Turba rubia con perlita o arena de cuarzo; también musgo sphagnum.",
            "Dormancia": "Depende de la especie: las tropicales crecen todo el año; las tuberosas y algunas de clima frío descansan.",
            "Alimentación": "Captura insectos pequeños por sí sola; en interior puede alimentarse con moscas de la fruta o comida de peces en polvo.",
        },
        video: "",
    },
    "Sarracenia": {
        common: "Planta jarra norteamericana",
        trap: "Trampa de caída (jarra erguida)",
        color: "#eb6834",
        text: "#ffffff",
        hint: "Jarras erguidas que nacen del rizoma, con un opérculo o tapa sobre la boca y venas de colores que atraen a los insectos.",
        care: {
            "Luz": "Sol pleno; es de las carnívoras que más luz necesita para colorear y mantener las jarras rectas.",
            "Agua": "Agua destilada o de lluvia; bandeja con 2 a 4 cm de agua en crecimiento, menos en invierno.",
            "Sustrato": "Turba rubia con perlita; macetas altas porque el rizoma y las raíces son profundos.",
            "Dormancia": "Reposo invernal obligatorio; las jarras se secan y vuelven a brotar en primavera.",
            "Alimentación": "No necesita ayuda; las jarras se llenan de insectos solas.",
        },
        video: "",
    },
    "Nepenthes": {
        common: "Planta jarra tropical",
        trap: "Trampa de caída (jarro colgante)",
        color: "#8e44ad",
        text: "#ffffff",
        hint: "Jarros que cuelgan del extremo de un zarcillo al final de cada hoja, con peristoma resbaloso y líquido digestivo en el fondo.",
        care: {
            "Luz": "Luz brillante pero filtrada; el sol directo del mediodía quema las hojas.",
            "Agua": "Agua destilada o de lluvia por arriba, manteniendo el sustrato húmedo pero nunca encharcado; sin bandeja permanente.",
            "Sustrato": "Musgo sphagnum con perlita o corteza; muy aireado.",
            "Humedad": "Alta (más del 60 %); si baja, la planta deja de formar jarros. Las de tierras altas necesitan noches frescas; las de tierras bajas, calor constante.",
            "Alimentación": "Opcional: un insecto o un poco de abono muy diluido dentro de los jarros, nunca en el sustrato.",
        },
        video: "",
    },
    "Darlingtonia": {
        common: "Planta cobra",
        trap: "Trampa de caída (capucha)",
        color: "#16a085",
        text: "#ffffff",
        hint: "Jarras que se curvan en una cabeza abombada con ventanas translúcidas y una lengua bífida bajo la boca, como una cobra erguida.",
        care: {
            "Luz": "Sol o luz muy brillante en la parte aérea, pero con las raíces frescas.",
            "Agua": "Agua destilada fría; lo ideal es riego frecuente por arriba o agua en movimiento para enfriar el sustrato.",
            "Sustrato": "Musgo sphagnum vivo con perlita o piedra pómez; drenaje alto.",
            "Dormancia": "Reposo invernal; en clima templado agradece noches frías todo el año.",
            "Clave": "Es la más exigente del grupo: raíces calientes o agua estancada la matan rápido.",
        },
        video: "",
    },
    "Heliamphora": {
        common: "Jarra de sol de los tepuyes",
        trap: "Trampa de caída (jarra con cuchara de néctar)",
        color: "#d4a017",
        text: "#1a1a1a",
        hint: "Jarras en forma de embudo sin tapa completa, coronadas por una pequeña cuchara de néctar; originarias de las cumbres de los tepuyes de Venezuela.",
        care: {
            "Luz": "Muy alta; con luz artificial potente o sol filtrado las jarras se ponen rojas.",
            "Agua": "Agua destilada o de lluvia, sustrato siempre húmedo y con buen drenaje.",
            "Sustrato": "Musgo sphagnum con perlita; macetas con muchos agujeros.",
            "Humedad": "Alta, con noches frescas (10 a 18 °C) y días templados; no soporta el calor seco.",
            "Dormancia": "No tiene; crece todo el año si las condiciones se mantienen.",
        },
        video: "",
    },
    "Pinguicula": {
        common: "Grasilla",
        trap: "Trampa adhesiva (hoja pegajosa)",
        color: "#2a78d6",
        text: "#ffffff",
        hint: "Roseta de hojas carnosas cubiertas de una capa pegajosa casi invisible; parece una suculenta y da flores llamativas sobre tallos altos.",
        care: {
            "Luz": "Brillante, con algo de sol suave; en interior crece bien bajo luz artificial.",
            "Agua": "Agua destilada o de lluvia; menos que otras carnívoras. Las especies mexicanas pasan el invierno con hojas pequeñas y suculentas y casi sin riego.",
            "Sustrato": "Mineral y aireado: perlita, vermiculita, piedra pómez o arena con poca turba.",
            "Humedad": "Media; tolera ambientes de interior.",
            "Alimentación": "Captura mosquitos y moscas de la fruta por sí sola; es útil contra plagas pequeñas.",
        },
        video: "",
    },
    "No carnivora": {
        common: "Otra planta u objeto",
        trap: "Sin trampa",
        color: "#5d6d7e",
        text: "#ffffff",
        hint: "La imagen no parece una de las siete carnívoras que reconoce el modelo: puede ser una suculenta, una orquídea, una bromelia, musgo o cualquier otra cosa.",
        care: {
            "Qué hacer": "Si crees que sí es una carnívora, intenta con una foto más cercana, con la trampa centrada y bien iluminada.",
            "Alcance": "El modelo solo distingue Dionaea, Drosera, Sarracenia, Nepenthes, Darlingtonia, Heliamphora y Pinguicula. Otras carnívoras (Cephalotus, Utricularia, Drosophyllum...) caen aquí o en el género más parecido.",
        },
        video: "",
    },
};
