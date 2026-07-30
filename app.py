import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# Configuración de la página Streamlit
st.set_page_config(
    page_title="Clasificación de imágenes - Reciclaje",
    page_icon="♻️",
    layout="centered",
)

# Diccionario de traducción a español
LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura",
}


@st.cache_resource
def load_recycling_model():
    """Carga el modelo y las clases desde los archivos locales."""
    try:
        model = tf.keras.models.load_model("waste_mobilenet.keras")
        with open("class_names.json", "r") as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Error al cargar el modelo o el archivo de clases: {e}")
        return None, None


model, class_names = load_recycling_model()

st.title("Clasificación de imágenes - Reciclaje - Servicio en la nube - Astrid Castellanos")
st.write(
    "Suba una imagen de un residuo para clasificarlo con el modelo de visión por computadora."
)

uploaded_file = st.file_uploader(
    "Seleccione una imagen", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    # Cargar y visualizar imagen subida
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesamiento: Ajuste proporcional y recorte central (evita deformación)
    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Inferencia / Predicción
    preds = model.predict(img_array)[0]
    
    # Obtener el índice con mayor probabilidad
    top_class_idx = np.argmax(preds)
    predicted_label = class_names[top_class_idx]
    label_es = LABELS_ES.get(predicted_label, predicted_label)
    confidence = preds[top_class_idx] * 100

    # Resultado principal
    st.success(f"**Resultado principal:** {label_es} ({confidence:.2f}% de confianza)")

    # Detección de ambigüedad entre clases transparentes/similares
    sorted_indices = np.argsort(preds)[::-1]
    top1_idx = sorted_indices[0]
    top2_idx = sorted_indices[1]
    
    label_1 = class_names[top1_idx]
    label_2 = class_names[top2_idx]
    
    # Si la diferencia entre Vidrio y Plástico es muy estrecha o la confianza es baja
    if {label_1, label_2} == {"glass", "plastic"}:
        prob_1 = preds[top1_idx] * 100
        prob_2 = preds[top2_idx] * 100
        if abs(prob_1 - prob_2) < 25.0:
            st.info(
                f"💡 **Nota de transparencia:** Se detectaron características compatibles con "
                f"**{LABELS_ES.get(label_1, label_1)}** ({prob_1:.1f}%) y "
                f"**{LABELS_ES.get(label_2, label_2)}** ({prob_2:.1f}%). "
                f"Los materiales translúcidos con reflejos de luz suelen compartir características visuales cercanas."
            )

    # Desglose de Top 3 predicciones
    st.subheader("Top 3 Predicciones:")
    top3_idx = sorted_indices[:3]
    for idx in top3_idx:
        raw_name = class_names[idx]
        translated = LABELS_ES.get(raw_name, raw_name)
        st.write(f"- **{translated}**: {preds[idx]*100:.2f}%")

elif uploaded_file is None:
    st.info("Cargue una imagen para iniciar la clasificación.")
