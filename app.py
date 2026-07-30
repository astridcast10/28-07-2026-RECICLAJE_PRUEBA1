import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Configuración de página con ícono en la pestaña/logo
st.set_page_config(
    page_title="Clasificación de imágenes - Reciclaje",
    page_icon="♻️",
    layout="wide",
)

# Estilos CSS con paleta colorida pero limpia
st.markdown("""
    <style>
    /* Fondo general */
    .stApp {
        background-color: #f7f9fc;
    }
    
    /* Encabezado principal */
    .main-header {
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: #ffffff !important;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #d1d8e0;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Tarjeta del Resultado Principal */
    .result-badge-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 12px rgba(56, 239, 125, 0.25);
        margin-bottom: 1.5rem;
    }
    .result-badge-card label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #e0f2fe;
        font-weight: 600;
        display: block;
        margin-bottom: 0.3rem;
    }
    .result-badge-card h2 {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }

    /* Tarjetas del Top 3 (Diseño por filas contrastadas) */
    .top-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 5px solid #4e4376;
    }
    .top-card-1 { border-left-color: #38ef7d; }
    .top-card-2 { border-left-color: #3b82f6; }
    .top-card-3 { border-left-color: #f59e0b; }

    .top-title {
        font-weight: 600;
        color: #1e293b;
        font-size: 1.05rem;
    }
    .top-badge {
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-1 { background-color: #d1fae5; color: #065f46; }
    .badge-2 { background-color: #dbeafe; color: #1e40af; }
    .badge-3 { background-color: #fef3c7; color: #92400e; }
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


# Banner Superior
st.markdown("""
    <div class="main-header">
        <h1>Clasificación de imágenes - Reciclaje</h1>
        <p>Servicio en la nube — Astrid Castellanos</p>
    </div>
""", unsafe_allow_html=True)

model, class_names = load_recycling_model()

# Subida de imagen
uploaded_file = st.file_uploader(
    "Seleccione una imagen para clasificar", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col_left, col_right = st.columns([1, 1], gap="large")

    image = Image.open(uploaded_file).convert("RGB")

    with col_left:
        st.markdown("### Imagen cargada")
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

    with col_right:
        st.markdown("### Resultado del Análisis")
        
        # Tarjeta verde principal
        st.markdown(f"""
            <div class="result-badge-card">
                <label>Material Predominante</label>
                <h2>{label_es}</h2>
            </div>
        """, unsafe_allow_html=True)

        st.write(f"**Nivel de Confianza:** `{confidence:.2f}%`")
        st.progress(float(confidence / 100))

        st.markdown("<br>", unsafe_allow_html=True)

        # Top 3 Predicciones con colores independientes
        st.markdown("### Top 3 Predicciones")
        top3_idx = np.argsort(preds)[-3:][::-1]
        
        styles = [
            ("top-card-1", "badge-1"),
            ("top-card-2", "badge-2"),
            ("top-card-3", "badge-3"),
        ]

        for i, idx in enumerate(top3_idx):
            raw_name = class_names[idx]
            translated = LABELS_ES.get(raw_name, raw_name)
            prob = preds[idx] * 100
            card_style, badge_style = styles[i]
            
            st.markdown(f"""
                <div class="top-card {card_style}">
                    <span class="top-title">{i+1}. {translated}</span>
                    <span class="top-badge {badge_style}">{prob:.2f}%</span>
                </div>
            """, unsafe_allow_html=True)

else:
    st.info("Cargue una imagen para iniciar la clasificación.")
