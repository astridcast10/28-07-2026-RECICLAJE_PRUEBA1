import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Clasificación de imágenes - Reciclaje",
    page_icon="♻️",
    layout="centered",
)

# Traducción a español
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
    model = tf.keras.models.load_model("waste_mobilenet.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_recycling_model()

st.title("Clasificación de imágenes - Reciclaje - Servicio en la nube - Astrid Castellanos")
st.write(
    "Suba una imagen **de un solo objeto centrado** para clasificarla con el modelo MobileNetV2."
)

uploaded_file = st.file_uploader(
    "Seleccione una imagen", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesamiento ajustado: Recorte central para evitar deformar la imagen
    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    preds = model.predict(img_array)[0]
    
    # Obtener las 2 clases más probables
    top2_idx = np.argsort(preds)[-2:][::-1]
    top1_idx, top2_idx_val = top2_idx[0], top2_idx[1]
    
    label_1 = class_names[top1_idx]
    label_2 = class_names[top2_idx_val]
    
    prob_1 = preds[top1_idx] * 100
    prob_2 = preds[top2_idx_val] * 100

    st.success(f"**Resultado principal:** {LABELS_ES.get(label_1, label_1)} ({prob_1:.2f}% de confianza)")

    # Si confunde materiales transparentes (Vidrio vs Plástico)
    if {label_1, label_2} == {"glass", "plastic"}:
        st.info(
            "💡 **Sugerencia:** Los materiales transparentes como el plástico PET y el vidrio "
            "tienen reflejos visuales muy parecidos. Para una mejor predicción, procura que se vea la tapa o el contenedor completo."
        )

    # Top 3
    st.subheader("Top 3 Predicciones:")
    top3_idx = np.argsort(preds)[-3:][::-1]
    for idx in top3_idx:
        raw_name = class_names[idx]
        translated = LABELS_ES.get(raw_name, raw_name)
        st.write(f"- **{translated}**: {preds[idx]*100:.2f}%")
else:
    st.info("Cargue una imagen para iniciar la clasificación.")
