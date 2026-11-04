import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Big Five Predictor & Cluster by Gerry", layout="wide")
st.title("🧠 Clasificador de Personalidad Big Five (OCEAN)")
st.write("Ingresa tus resultados del test Big Five para descubrir tu cluster de personalidad, famosos con rasgos similares y comparar tus puntuaciones en un gráfico de radar.")

# 1. Cargar el dataset, modelo y escalador
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv"
    df = pd.read_csv(url)

    # Crear mapeo de categorías
    mapping_categorias = {
        1: 'Actor/actriz', 2: 'Cantante', 3: 'Modelo', 4: 'Tv, series',
        5: 'Radio', 6: 'Tecnología', 7: 'Deportes', 8: 'Politica', 9: 'Escritor'
    }
    df['categoria_nombre'] = df['categoria'].map(mapping_categorias)
    return df

@st.cache_resource
def load_models():
    kmeans = joblib.load('kmeans_bigFive_model.joblib')
    scaler = joblib.load('scaler_bigFive.joblib')
    return kmeans, scaler

try:
    df = load_data()
    kmeans, scaler = load_models()
except Exception as e:
    st.error(f"Error al cargar los datos o modelos: {e}")
    st.stop()

# 2. Sidebar para entrada de datos
st.sidebar.header("Tus Resultados Big Five")
op_val = st.sidebar.slider("Apertura a la experiencia (op)", 0.0, 100.0, 67.8, 0.1)
co_val = st.sidebar.slider("Responsabilidad / Orden (co)", 0.0, 100.0, 38.2, 0.1)
ex_val = st.sidebar.slider("Extraversión (ex)", 0.0, 100.0, 43.0, 0.1)
ag_val = st.sidebar.slider("Amabilidad / Empatía (ag)", 0.0, 100.0, 62.9, 0.1)
e_val = st.sidebar.slider("Neuroticismo (ne)", 0.0, 100.0, 68.0, 0.1)

# 3. Procesar Predicción
user_data = pd.DataFrame([{ 'op': op_val, 'co': co_val, 'ex': ex_val, 'ag': ag_val, 'ne': e_val }])
user_scaled = scaler.transform(user_data)
predicted_cluster = kmeans.predict(user_scaled)[0]

# 4. Mostrar Resultados en Pantalla Principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"🎯 Cluster Asignado: **Cluster {predicted_cluster}**")
    st.metric(label="Tu Cluster", value=f"Cluster {predicted_cluster}")

    # Obtener famosos de este cluster
    famosos_cluster = df[df['cluster'] == predicted_cluster]
    st.write(f"### 🌟 Famosos en tu mismo cluster ({len(famosos_cluster)} en total)")
    st.dataframe(famosos_cluster[['usuario', 'categoria_nombre', 'op', 'co', 'ex', 'ag', 'ne']].head(5))

with col2:
    st.write("### 📊 Comparativa de Radar")

    # Rasgos de interés
    features = ['op', 'co', 'ex', 'ag', 'ne']
    num_vars = len(features)

    # Obtener promedios del cluster asignado
    cluster_means = famosos_cluster[features].mean().values.tolist()
    cluster_means += cluster_means[:1]

    # Preparar datos del usuario
    user_values = [op_val, co_val, ex_val, ag_val, e_val]
    user_values += user_values[:1]

    # Ángulos de la gráfica polar
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Dibujar gráfico
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Promedio Cluster
    ax.plot(angles, cluster_means, color='#1aafbc', linewidth=2, label=f'Promedio Cluster {predicted_cluster}')
    ax.fill(angles, cluster_means, color='#1aafbc', alpha=0.25)

    # Valores del Usuario
    ax.plot(angles, user_values, color='#e74c3c', linewidth=2, linestyle='dashed', label='Tus Resultados')
    ax.fill(angles, user_values, color='#e74c3c', alpha=0.15)

    # Estética del radar
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), features)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)
