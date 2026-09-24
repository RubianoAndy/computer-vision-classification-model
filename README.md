# Clasificador de residuos · Orgánico vs Reciclable

Proyecto final de **Visión por Computador** (Maestría en Inteligencia Artificial, Universidad de La Salle, 2026).
Un clasificador binario de residuos entrenado en **Teachable Machine**, publicado en la nube de la
plataforma e integrado en una **aplicación web** que sugiere la caneca correcta según el código de
colores colombiano (Resolución 2184 de 2019): **verde** para orgánicos y **blanca** para reciclables.

| Recurso | Enlace |
|---|---|
| Modelo en la nube (endpoint) | https://teachablemachine.withgoogle.com/models/UerwbSsVX/ |
| Dataset | [techsash/waste-classification-data](https://www.kaggle.com/datasets/techsash/waste-classification-data) |
| Herramienta | [Teachable Machine](https://teachablemachine.withgoogle.com/) |
| Informe | carpeta hermana `computer vision-classification-model-report` |

---

## Resultados en una mirada

Sobre **500 imágenes de prueba** que el modelo nunca vio (250 por clase), evaluadas desde el endpoint:

| Métrica | Valor | IC 95 % |
|---|---|---|
| Exactitud | **0,844** | [0,812; 0,874] |
| F1 macro | 0,842 | [0,809; 0,872] |
| AUC | 0,957 | [0,940; 0,972] |
| Pérdida logarítmica | 0,782 | |

| Clase | Precisión | Exhaustividad | F1 |
|---|---|---|---|
| Orgánico | 0,781 | 0,956 | 0,860 |
| Reciclable | 0,943 | 0,732 | 0,824 |

Matriz de confusión (filas = real, columnas = predicha): Orgánico `239 / 11`, Reciclable `67 / 183`.
El modelo sesga hacia *Orgánico*: 67 de los 78 errores son reciclables enviados a la caneca verde.
Teachable Machine reporta 0,907 en su prueba interna, 6,3 puntos por encima de la independiente.

---

## Estructura

```
.
├── app/                      # Aplicación web (HTML + CSS + JS, sin build)
│   ├── index.html
│   ├── styles.css
│   ├── app.js                # Carga del modelo, tres modos, métricas en vivo
│   └── config.js             # MODEL_URL (endpoint) y descripción de las clases
├── scripts/
│   ├── dataset-inspection.py # Formatos, modos y tamaños del dataset
│   └── split-construction.py # dHash, exclusión de repetidos y muestreo 500/250 por clase
├── eval/
│   ├── model-inference.js    # Inferencia (nube o disco) -> CSV de probabilidades
│   ├── model-evaluation.py   # Métricas, ROC, AUC, bootstrap y figuras del informe
│   ├── error-gallery.py      # Galería de errores y umbral alternativo
│   ├── cloud-predictions.csv # Probabilidades del modelo en la nube (500 filas)
│   ├── cloud-metrics.json    # Métricas completas
│   └── local-predictions.csv # Mismas probabilidades con la copia descargada
├── models/
│   ├── tm-waste-model.zip    # Exportación TensorFlow.js descargada de Teachable Machine
│   └── tm-waste-model/       # model.json · weights.bin · metadata.json
├── segments/                 # (no versionado) train/ y test/ con Organico/ y Reciclable/
└── dataset/                  # (no versionado) caché de kagglehub
```

---

## Reproducir

### 1 · Dataset y segmentos

```bash
KAGGLEHUB_CACHE=./dataset python -c "import kagglehub; kagglehub.dataset_download('techsash/waste-classification-data')"
python scripts/dataset-inspection.py
python scripts/split-construction.py     # crea segments/train y segments/test
```

`split-construction.py` calcula un dHash por imagen, descarta las repetidas, excluye del
segmento de prueba las imágenes que también aparecen en TRAIN (**295** en el dataset original)
y toma 500 por clase para entrenar y 250 por clase para probar (semilla 42).

### 2 · Entrenamiento y despliegue en Teachable Machine

1. Proyecto de imagen con dos clases, **Orgánico** y **Reciclable**, y subir `segments/train/<clase>`.
2. *Preparar modelo* con los valores por defecto (50 épocas, lote 16, tasa 0,001).
3. *Exportar modelo → TensorFlow.js → Subir (enlace para compartir)*: la plataforma devuelve la URL
   del endpoint, que sirve `model.json`, `weights.bin` y `metadata.json`.
4. Opcional: *Descargar* para conservar una copia local en `models/`.

### 3 · Aplicación web

```bash
cd app
python -m http.server 8765
```

Abrir <http://127.0.0.1:8765/>. La app carga el modelo desde `MODEL_URL` (`config.js`) con
`@teachablemachine/image` y ejecuta la inferencia en el navegador. Modos:

- **Una imagen**: arrastrar o elegir una foto; muestra clase, caneca, confianza y probabilidades.
- **Conjunto de imágenes**: cargar una carpeta; si tiene subcarpetas `Orgánico/` y `Reciclable/`
  calcula exactitud, precisión, exhaustividad, F1 y matriz de confusión, y exporta un CSV.
- **Cámara**: clasificación en vivo desde la webcam.

### 4 · Evaluación fuera de la herramienta

```bash
cd eval
npm install
node model-inference.js https://teachablemachine.withgoogle.com/models/UerwbSsVX/ ../segments/test cloud-predictions.csv
python model-evaluation.py cloud-predictions.csv cloud
python error-gallery.py cloud-predictions.csv cloud
```

Las figuras se escriben en `../computer vision-classification-model-report/assets/images/results/`.
Para evaluar la copia local: `node model-inference.js ../models/tm-waste-model ../segments/test local-predictions.csv`.

---

## Requisitos

- Python 3.12 con `kagglehub`, `Pillow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`.
- Node.js 18+ (`@tensorflow/tfjs`, `jpeg-js`, instalados con `npm install` en `eval/`).
- Navegador moderno para la aplicación; conexión a Internet para descargar el modelo desde el endpoint.

---

## Autor

**Andrés Giovanny Rubiano Muñoz** · arubiano67@unisalle.edu.co
Maestría en Inteligencia Artificial · Universidad de La Salle · Bogotá, Colombia

Las imágenes provienen del dataset *Waste Classification data* publicado en Kaggle por Sashaank Sekar;
no se redistribuyen en este repositorio y su uso es exclusivamente académico.
