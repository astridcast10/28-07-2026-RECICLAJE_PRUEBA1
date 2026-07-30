import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración de página
st.set_page_config(
    page_title="Clasificación de imágenes - Reciclaje",
    page_icon="♻️",
    layout="wide",
)

# Estilos CSS personalizados para mejorar el diseño de la interfaz
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white !important;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #e0e0e0;
        font-size: 1.1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

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


# Encabezado principal
st.markdown("""
    <div class="main-header">
        <h1>♻️ Clasificador Intelligente de Residuos</h1>
        <p>Servicio en la nube | Astrid Castellanos</p>
    </div>
""", unsafe_allow_html=True)

model, class_names = load_recycling_model()

# Panel de subida
st.write("### 📤 Cargar Imagen")
uploaded_file = st.file_uploader(
    "Seleccione una imagen para clasificar el material de reciclaje", 
    type=["jpg", "jpeg", "png"]
)

st.divider()

if uploaded_file is not None:
    # Organización en dos columnas para el resultado
    col1, col2 = st.columns([1, 1], gap="large")

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.subheader("🖼️ Imagen Analizada")
        st.image(image, use_column_width=True)

    # Preprocesamiento
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    preds = model.predict(img_array)[0]
    top_class_idx = np.argmax(preds)
    predicted_label = class_names[top_class_idx]
    label_es = LABELS_ES.get(predicted_label, predicted_label)
    confidence = preds[top_class_idx] * 100

    with col2:
        st.subheader("📊 Resultados del Análisis")
        
        # Tarjeta destacada del resultado principal
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0; color:#555;">Material Detectado:</h4>
                <h2 style="margin:0; color:#2e7d32; font-size: 2.2rem;">{label_es}</h2>
            </div>
        """, unsafe_allow_html=True)

        # Barra de progreso visual para la confianza
        st.write(f"**Nivel de Confianza:** {confidence:.2f}%")
        st.progress(float(confidence / 100))

        st.space(2)

        # Sección del Top 3
        st.subheader("🏆 Top 3 Predicciones:")
        top3_idx = np.argsort(preds)[-3:][::-1]
        
        for idx in top3_idx:
            raw_name = class_names[idx]
            translated = LABELS_ES.get(raw_name, raw_name)
            prob = preds[idx] * 100
            
            # Formato de métrica secundaria limpia
            col_label, col_val = st.columns([2, 1])
            with col_label:
                st.write(f"**{translated}**")
            with col_val:
                st.write(f"`{prob:.2f}%`")

else:
    # Mensaje inicial cuando no hay archivo
    st.info("👋 Por favor, carga una imagen en formato JPG o PNG para iniciar el análisis del residuo.")
