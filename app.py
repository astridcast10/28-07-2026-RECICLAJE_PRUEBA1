import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración de página amplia
st.set_page_config(
    page_title="Clasificación de imágenes - Reciclaje",
    layout="wide",
)

# Estilos CSS personalizados (Paleta de colores limpia y moderna)
st.markdown("""
    <style>
    /* Estilo del contenedor principal */
    .stApp {
        background-color: #f4f6f9;
    }
    
    /* Encabezado superior */
    .header-box {
        background: linear-gradient(135deg, #0d6efd 0%, #0d3b66 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .header-box h1 {
        color: #ffffff !important;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .header-box p {
        color: #e0e8f5;
        font-size: 1.1rem;
        margin-bottom: 0;
    }

    /* Tarjeta del resultado principal */
    .result-card {
        background: linear-gradient(135deg, #198754 0%, #115c39 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(25, 135, 84, 0.2);
        margin-bottom: 1.5rem;
    }
    .result-card h3 {
        color: #d1e7dd !important;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .result-card h2 {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }

    /* Tarjetas fijas para el Top 3 */
    .top-item {
        background-color: #ffffff;
        border-left: 6px solid #0d6efd;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 0.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .top-item-title {
        font-weight: 600;
        color: #212529;
        font-size: 1.05rem;
    }
    .top-item-badge {
        background-color: #e7f1ff;
        color: #0d6efd;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
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


# Banner de bienvenida
st.markdown("""
    <div class="header-box">
        <h1>Clasificador Inteligente de Residuos</h1>
        <p>Servicio de Visión por Computadora en la Nube — Astrid Castellanos</p>
    </div>
""", unsafe_allow_html=True)

model, class_names = load_recycling_model()

# Selector de archivo
uploaded_file = st.file_uploader(
    "Cargue una imagen para analizar", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Organización en 2 columnas principales
    col_img, col_results = st.columns([1, 1], gap="large")

    image = Image.open(uploaded_file).convert("RGB")

    with col_img:
        st.markdown("#### Vista previa")
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

    with col_results:
        st.markdown("#### Resultado del Diagnóstico")
        
        # Tarjeta principal en verde azulado
        st.markdown(f"""
            <div class="result-card">
                <h3>Categoría Detectada</h3>
                <h2>{label_es}</h2>
            </div>
        """, unsafe_allow_html=True)

        # Medidor de confianza
        st.write(f"**Confianza de predicción:** `{confidence:.2f}%`")
        st.progress(float(confidence / 100))

        st.markdown("---")

        # Tarjetas secundarias para el Top 3
        st.markdown("#### Distribución de Probabilidades")
        top3_idx = np.argsort(preds)[-3:][::-1]
        
        for idx in top3_idx:
            raw_name = class_names[idx]
            translated = LABELS_ES.get(raw_name, raw_name)
            prob = preds[idx] * 100
            
            st.markdown(f"""
                <div class="top-item">
                    <span class="top-item-title">{translated}</span>
                    <span class="top-item-badge">{prob:.2f}%</span>
                </div>
            """, unsafe_allow_html=True)

else:
    st.info("Seleccione una imagen en formato JPG, JPEG o PNG para comenzar.")
