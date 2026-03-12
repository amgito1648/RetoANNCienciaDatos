import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. Configuración de la interfaz
st.set_page_config(page_title="Predictor de Crédito AI", page_icon="💳")

@st.cache_resource
def load_assets():
    # Carga de los 5 archivos generados por tu script de entrenamiento
    model = load_model('modelo_credito.h5')
    scaler = joblib.load('scaler.bin')
    selector = joblib.load('selector.bin')
    pca = joblib.load('pca_v2.bin')
    features = joblib.load('todas_las_features.bin')
    return model, scaler, selector, pca, features

# Cargar todo
try:
    model, scaler, selector, pca, all_features = load_assets()
except Exception as e:
    st.error(f"Error al cargar archivos: {e}. Asegúrate de subirlos a GitHub.")
    st.stop()

st.title("💳 Sistema de Clasificación de Riesgo")
st.write("Complete la información del cliente para obtener la predicción de riesgo.")

# 2. Formulario de entrada
with st.form("form_prediccion"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Edad", value=30)
        income = st.number_input("Ingreso Anual", value=50000.0)
        monthly = st.number_input("Salario Mensual en mano", value=4000.0)
        accounts = st.number_input("Número de Cuentas Bancarias", value=2)
        cards = st.number_input("Número de Tarjetas de Crédito", value=3)
        interest = st.number_input("Tasa de Interés (%)", value=12.0)
    with col2:
        loans = st.number_input("Número de Préstamos", value=2)
        delayed = st.number_input("Pagos Atrasados", value=1)
        inquiries = st.number_input("Consultas de Crédito", value=2)
        debt = st.number_input("Deuda Pendiente Total", value=1000.0)
        utilization = st.slider("Ratio de Utilización de Crédito", 0.0, 100.0, 30.0)
        mix = st.selectbox("Mix de Crédito", options=["Good", "Standard", "Bad"])
    
    submit = st.form_submit_button("Analizar Perfil")

# 3. Lógica de ejecución
if submit:
    try:
        # A. Crear DataFrame base con ceros (asegura coincidencia con all_features)
        input_df = pd.DataFrame(np.zeros((1, len(all_features))), columns=all_features)
        
        # B. Llenar los campos que el usuario ingresó
        mapping = {
            'Age': age,
            'Annual_Income': income,
            'Monthly_Inhand_Salary': monthly,
            'Num_Bank_Accounts': accounts,
            'Num_Credit_Card': cards,
            'Interest_Rate': interest,
            'Num_of_Loan': loans,
            'Num_of_Delayed_Payment': delayed,
            'Num_Credit_Inquiries': inquiries,
            'Outstanding_Debt': debt,
            'Credit_Utilization_Ratio': utilization
        }

        for col, val in mapping.items():
            if col in input_df.columns:
                input_df.at[0, col] = val

        # C. Manejo de Credit_Mix (Ordinal o Dummy)
        mix_val = {'Bad': 0, 'Standard': 1, 'Good': 2}[mix]
        if 'Credit_Mix' in input_df.columns:
            input_df.at[0, 'Credit_Mix'] = mix_val
        
        # Si el modelo espera dummies (ej: Credit_Mix_Standard)
        dummy_col = f"Credit_Mix_{mix}"
        if dummy_col in input_df.columns:
            input_df.at[0, dummy_col] = 1

        # D. Seguro contra NaNs (por si el scaler espera algo que no enviamos)
        input_df = input_df.fillna(0.0)

        # E. Pipeline de transformación
        # 1. Escalar (debe recibir el número exacto de columnas originales)
        X_scaled = scaler.transform(input_df)
        
        # 2. Seleccionar mejores características
        X_kbest = selector.transform(X_scaled)
        
        # 3. Reducción PCA
        X_pca = pca.transform(X_kbest)
        
        # F. Inferencia
        prediction = model.predict(X_pca)
        clase_final = np.argmax(prediction)
        probabilidad = np.max(prediction) * 100

        # G. Resultados visuales
        etiquetas = {0: "POBRE 🔴", 1: "ESTÁNDAR 🟡", 2: "BUENO 🟢"}
        st.markdown("---")
        st.subheader(f"Resultado de la Evaluación: **{etiquetas[clase_final]}**")
        st.progress(int(probabilidad))
        st.write(f"Nivel de confianza del análisis: {probabilidad:.2f}%")

    except Exception as e:
        st.error(f"Error técnico durante la predicción: {e}")
        st.info("Verifica que los archivos .bin y .h5 en GitHub correspondan a la última versión del entrenamiento.")