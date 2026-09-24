<div align="center">
    <img src="assets/images/Logo.png" width="250" alt="Logo Universidad de La Salle">
</div>

# Clasificador de residuos orgánicos y reciclables con Teachable Machine, desplegado en la nube e integrado en una aplicación web para la separación en la fuente

## 📋 Información General

<div align="center">
    <img src="assets/images/author/Andy%20Rubiano.png" width="180" alt="Andrés Giovanny Rubiano Muñoz" style="border-radius: 10px;">
</div>

| Aspecto | Detalles |
|--------|----------|
| **Autor** | Andrés Giovanny Rubiano Muñoz "Andy Rubiano" |
| **Correo** | arubiano67@unisalle.edu.co |
| **Asignatura** | Visión por Computador |
| **Docente** | Carlos Alejandro Espejo Villarraga |
| **Actividad** | Actividad 6 · Proyecto Final – Implementemos un modelo de clasificación |
| **Programa** | Maestría en Inteligencia Artificial |
| **Universidad** | Universidad de La Salle |
| **Entregable** | Modelo en la nube + aplicación web + scripts de evaluación (el informe está en `../computer vision-classification-model-report`) |
| **Año** | 2026 |

---

## 🎯 Descripción

Este repositorio contiene el **proyecto** del informe: los scripts que preparan el dataset, la aplicación web que consume el modelo y los scripts que lo evalúan. El modelo es un clasificador binario de residuos, *orgánico* o *reciclable*, entrenado en **Teachable Machine** y publicado en la nube de la plataforma, pensado como asistente de **separación en la fuente** según el código de colores colombiano (Resolución 2184 de 2019): caneca **verde** para orgánicos y caneca **blanca** para reciclables.

| Recurso | Enlace |
|---|---|
| **Repositorio en GitHub** | https://github.com/RubianoAndy/computer-vision-classification-model |
| **Aplicación publicada (GitHub Pages)** | https://rubianoandy.github.io/computer-vision-classification-model/ |
| **Modelo en la nube (endpoint)** | https://teachablemachine.withgoogle.com/models/UerwbSsVX/ |
| **Dataset** | [techsash/waste-classification-data](https://www.kaggle.com/datasets/techsash/waste-classification-data) |
| **Herramienta** | [Teachable Machine](https://teachablemachine.withgoogle.com/) |
| **Informe** | `../computer vision-classification-model-report/build/main.pdf` |

> ℹ️ **Por qué la inferencia ocurre en el navegador.** El endpoint de Teachable Machine no es una API de predicción sino un servidor de archivos estáticos: entrega `model.json`, `weights.bin` y `metadata.json`, y quien los descarga ejecuta el modelo con TensorFlow.js. No hay costo por consulta y las fotografías nunca salen del dispositivo del usuario.

---

## 🧪 Proceso

### 1 · Dataset y segmentos

25.077 imágenes en dos carpetas, *O* (orgánicos) y *R* (reciclables), descargadas con `kagglehub`. Con *dHash* se eliminaron los duplicados por contenido y se encontró que **295 imágenes de la partición TEST también están en TRAIN** (el 20,8 % de sus orgánicos). Los segmentos finales son 500 por clase para entrenar y 250 por clase para probar, sin ninguna imagen compartida.

| Clase | Partición | Originales | Únicas | Repetidas en TRAIN |
|---|---|---|---|---|
| Orgánico | TRAIN | 12.565 | 12.503 | — |
| Orgánico | TEST | 1.401 | 1.401 | 291 |
| Reciclable | TRAIN | 9.999 | 9.797 | — |
| Reciclable | TEST | 1.112 | 1.098 | 4 |

> ⚠️ Quien evalúe con la partición TEST tal como viene en Kaggle está midiendo en parte memoria y no generalización.

### 2 · Entrenamiento en Teachable Machine

Proyecto de imagen con las clases **Orgánico** y **Reciclable**, 500 imágenes cada una, con los hiperparámetros por defecto: 50 épocas, lote de 16 y tasa de aprendizaje de 0,001. La herramienta aparta el 15 % de cada clase como prueba interna, así que solo 850 imágenes ajustan los pesos.

### 3 · Despliegue en la nube

*Exportar modelo → TensorFlow.js → Subir (enlace para compartir)*. La plataforma aloja el modelo y devuelve la URL del endpoint. También se descargó una copia (`models/tm-waste-model.zip`, 2,1 MiB) que produce **exactamente las mismas probabilidades** que la versión publicada.

### 4 · Aplicación web

HTML + CSS + JavaScript sin *framework* de interfaz, con `@teachablemachine/image` y TensorFlow.js desde CDN. Tres modos:

| Modo | Qué hace |
|---|---|
| **Una imagen** | Arrastrar o elegir una foto; muestra la clase, la caneca con su color, la confianza y las dos probabilidades |
| **Conjunto de imágenes** | Cargar una carpeta completa; si trae subcarpetas `Orgánico/` y `Reciclable/` calcula exactitud, precisión, exhaustividad, F1 y matriz de confusión en vivo, y exporta un CSV |
| **Cámara** | Clasificación en tiempo real desde la webcam |

Los tres modos comparten un **interruptor de umbral de decisión**: *por defecto* (gana la clase con mayor probabilidad, umbral 0,5) o *ajustado* (solo se declara *Orgánico* si su probabilidad supera el umbral elegido en validación). En el modo *Conjunto de imágenes* el cambio recalcula al instante las métricas sin volver a ejecutar el modelo, lo que permite ver el efecto del ajuste sobre las mismas imágenes.

### 5 · Evaluación fuera de la herramienta

El panel *Más datos* de Teachable Machine no entrega la probabilidad de cada imagen, y sin ella no hay curva ROC. Por eso `eval/model-inference.js` carga el modelo **directamente desde el endpoint** en Node.js, repite el preprocesamiento de la plataforma (recorte central, 224 × 224, rango [-1, 1]) y escribe un CSV con las probabilidades; `eval/model-evaluation.py` calcula las métricas con scikit-learn y dibuja las figuras del informe.

---

## 📊 Resultados

### Lo que reporta Teachable Machine (prueba interna, 150 imágenes)

Orgánico 0,95 · Reciclable 0,87 · exactitud interna 0,907. La pérdida de prueba crece desde la época 12 mientras la de entrenamiento cae a cero: sobreajuste en la calibración, no en la etiqueta.

### Evaluación sobre el segmento independiente (500 imágenes)

| Métrica | Valor | IC 95 % (bootstrap) |
|---|---|---|
| **Exactitud** | **0,844** | [0,812; 0,874] |
| F1 macro | 0,842 | [0,809; 0,872] |
| AUC | 0,957 | [0,940; 0,972] |
| Pérdida logarítmica | 0,782 | |

| Clase | Precisión | Exhaustividad | F1 | Soporte |
|---|---|---|---|---|
| Orgánico | 0,781 | 0,956 | 0,860 | 250 |
| Reciclable | 0,943 | 0,732 | 0,824 | 250 |

Matriz de confusión (filas = real, columnas = predicha): Orgánico `239 / 11`, Reciclable `67 / 183`.

- **Sesgo hacia Orgánico**: 67 de los 78 errores son reciclables enviados a la caneca verde, el error menos costoso de los dos.
- **Ajuste del modelo**: el umbral de decisión se eligió sobre un segmento de validación independiente (0,977) y en prueba sube la exactitud de 0,844 a **0,888** (errores de 78 a 56, mejora pareada [0,020; 0,070]); está integrado en la app como interruptor.
- **Confianza**: media de 0,969, y de 0,908 en los errores; el modelo se equivoca con seguridad.
- **Interna vs. independiente**: la exactitud del panel, 0,907, queda 6,3 puntos por encima y fuera del intervalo de confianza.
- **Nube = local = app**: la copia descargada da las mismas probabilidades que el endpoint, y la app reporta 423 aciertos frente a 422 del script (una imagen de diferencia por el redimensionamiento del navegador).

Las figuras (matriz de confusión, curva ROC, histograma de probabilidades y galería de errores) se generan en `../computer vision-classification-model-report/assets/images/results/`.

---

## 📚 Estructura del Repositorio

La raíz está organizada para publicarse tal cual en **GitHub Pages**: `index.html` es la página, `src/` y `assets/` la alimentan, y todo lo que no forma parte de la aplicación vive en `utils/`.

```
.
├── index.html                        # La aplicación web: una imagen · conjunto de imágenes · cámara
├── README.md                         # Este archivo
├── .nojekyll                         # GitHub Pages sirve los archivos tal cual
├── .gitignore
├── src/                              # Lo que alimenta a index.html
│   ├── styles.css                    # Identidad Universidad de La Salle, formas cuadradas
│   ├── app.js                        # Carga del modelo, inferencia, métricas en vivo, CSV
│   └── config.js                     # MODEL_URL (endpoint) y descripción de las clases
├── assets/
│   └── images/
│       ├── Logo.png                  # Logo institucional (README y favicon)
│       ├── logo-white.png            # Logo en blanco para la barra y el pie de la app
│       └── author/                   # Fotografía del autor
└── utils/                            # Todo lo que no es la aplicación
    ├── scripts/
    │   ├── dataset-inspection.py     # Formatos, modos y tamaños del dataset
    │   └── split-construction.py     # dHash, exclusión de repetidos y muestreo 500/250 por clase
    ├── eval/
    │   ├── package.json              # @tensorflow/tfjs · jpeg-js
    │   ├── model-inference.js        # Inferencia (nube o disco) → CSV de probabilidades
    │   ├── model-evaluation.py       # Métricas, ROC, AUC, bootstrap y figuras del informe
    │   ├── threshold-selection.py    # Elige el umbral en validación y lo evalúa en prueba
    │   ├── error-gallery.py          # Galería de errores
    │   ├── cloud-predictions.csv     # Probabilidades en prueba (500 filas)
    │   ├── val-predictions.csv       # Probabilidades en validación (500 filas)
    │   ├── cloud-metrics.json        # Métricas con el umbral por defecto
    │   ├── cloud-threshold-metrics.json  # Umbral elegido y métricas con ambos umbrales
    │   └── local-predictions.csv     # Mismas probabilidades con la copia descargada
    └── models/
        ├── tm-waste-model.zip        # Exportación TensorFlow.js descargada de Teachable Machine
        └── tm-waste-model/           # model.json · weights.bin · metadata.json
```

Las carpetas `dataset/` (caché de kagglehub) y `segments/` (train/, val/ y test/) se crean en la raíz al correr los scripts y **no se versionan**.

> ℹ️ **Las imágenes no se versionan.** `dataset/` pesa unos 430 MB y `segments/` unos 80 MB; ambos se regeneran con los scripts de abajo, con semilla fija (`SEED = 42`), así que se obtienen exactamente las mismas 2.000 imágenes.

---

## ⚙️ Requisitos

| Componente | Dependencias |
|---|---|
| Scripts de Python (3.12) | `kagglehub`, `Pillow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib` |
| Evaluación en Node.js (18+) | `@tensorflow/tfjs`, `jpeg-js` (se instalan con `npm install` en `eval/`) |
| Aplicación web | Cualquier navegador moderno y conexión a Internet para descargar el modelo desde el endpoint |

---

## 🛠️ Reproducción

### 1 · Dataset y segmentos

```bash
KAGGLEHUB_CACHE=./dataset python -c "import kagglehub; kagglehub.dataset_download('techsash/waste-classification-data')"
python utils/scripts/dataset-inspection.py
python utils/scripts/split-construction.py
```

### 2 · Entrenamiento y despliegue

1. En Teachable Machine, proyecto de imagen con dos clases, **Orgánico** y **Reciclable**, y subir `segments/train/<clase>`.
2. *Preparar modelo* con los valores por defecto.
3. *Exportar modelo → TensorFlow.js → Subir (enlace para compartir)* y copiar la URL en `src/config.js` (`MODEL_URL`).
4. Opcional: *Descargar* para conservar una copia en `utils/models/`.

### 3 · Aplicación web

En local, desde la raíz del repositorio:

```bash
python -m http.server 8765
```

Abrir <http://127.0.0.1:8765/>. Para probar el modo *Conjunto de imágenes* con métricas, cargar la carpeta `segments/test`.

En **GitHub Pages** basta con publicar la rama desde la raíz (`/`): `index.html`, `src/` y `assets/` son todo lo que la página necesita, y el archivo `.nojekyll` evita que GitHub procese el sitio con Jekyll. El modelo se descarga desde el endpoint de Teachable Machine, así que no hay nada que compilar ni desplegar aparte.

### 4 · Evaluación

```bash
cd utils/eval
npm install
node model-inference.js https://teachablemachine.withgoogle.com/models/UerwbSsVX/ ../../segments/test cloud-predictions.csv
node model-inference.js https://teachablemachine.withgoogle.com/models/UerwbSsVX/ ../../segments/val val-predictions.csv
python model-evaluation.py cloud-predictions.csv cloud
python threshold-selection.py val-predictions.csv cloud-predictions.csv cloud
python error-gallery.py cloud-predictions.csv cloud
```

Para evaluar la copia local: `node model-inference.js ../models/tm-waste-model ../../segments/test local-predictions.csv`.

> ⚠️ Las carpetas de `segments/` van sin tilde (`Organico`, `Reciclable`) mientras que las clases del modelo la llevan (`Orgánico`). Los scripts y la app normalizan el nombre, así que no hay que renombrar nada.

---

## 🔑 Palabras Clave

`AUC` · `Clasificación de Imágenes` · `Curva ROC` · `Despliegue en la Nube` · `Exhaustividad` · `F1` · `Inteligencia Artificial` · `MobileNet` · `Precisión` · `Reciclaje` · `Separación en la Fuente` · `Teachable Machine` · `TensorFlow.js` · `Transferencia de Aprendizaje` · `Visión por Computador`

---

## 🔗 Enlaces Adicionales

- [Teachable Machine](https://teachablemachine.withgoogle.com/)
- [Biblioteca @teachablemachine/image](https://github.com/googlecreativelab/teachablemachine-community/tree/master/libraries/image)
- [TensorFlow.js](https://www.tensorflow.org/js)
- [Resolución 2184 de 2019 – Código de colores para la separación de residuos](https://www.minambiente.gov.co/)
- [Documentación de scikit-learn](https://scikit-learn.org/stable/)
- [Documentación de kagglehub](https://github.com/Kaggle/kagglehub)

---

## 📧 Contacto

**Andrés Giovanny Rubiano Muñoz**
Maestría en Inteligencia Artificial · Universidad de La Salle
arubiano67@unisalle.edu.co

---

## 📄 Derechos Reservados

© 2026 Andrés Giovanny Rubiano Muñoz (Andy Rubiano). Todos los derechos reservados.

Este trabajo académico y su contenido —investigación, código, metodologías y documentación— son propiedad intelectual conjunta de:

- **Andrés Giovanny Rubiano Muñoz** (Andy Rubiano) — Autor
- **Universidad de La Salle** — Institución académica

El uso, reproducción o distribución requiere autorización previa escrita de los titulares de derechos.

> ℹ️ Las imágenes provienen del dataset *Waste Classification data*, publicado por Sashaank Sekar en Kaggle. No se redistribuyen en este repositorio; su uso aquí es exclusivamente académico y se rige por los términos de la fuente original.

---

<div align="center">
  Universidad de La Salle | Bogotá D. C., Colombia
</div>
