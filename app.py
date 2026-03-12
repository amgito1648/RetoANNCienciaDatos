import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. Configuración de la página
st.set_page_config(page_title="Predictor de Crédito AI", page_icon="💳")

# CARGA DE ASSETS (Separados para evitar errores de tuplas)
@st.cache_resource
def get_model():
    return load_model('modelo_credito.h5')

@st.cache_resource
def get_transforms():
    sc = joblib.load('scaler.bin')
    sel = joblib.load('selector.bin')
    pc = joblib.load('pca.bin')
    feat = joblib.load('todas_las_features.bin')
    return sc, sel, pc, feat

# Asignación de variables
model = get_model()
scaler, selector, pca, all_features = get_transforms()

st.title("💳 Sistema de Clasificación de Riesgo")
st.write("Introduce los datos del cliente para evaluar su perfil crediticio.")

# 2. Formulario de entrada
with st.form("main_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (Edad)", value=30)
        income = st.number_input("Annual_Income", value=50000.0)
        monthly = st.number_input("Monthly_Inhand_Salary", value=4000.0)
        accounts = st.number_input("Num_Bank_Accounts", value=2)
        cards = st.number_input("Num_Credit_Card", value=3)
        interest = st.number_input("Interest_Rate", value=12.0)
    with col2:
        loans = st.number_input("Num_of_Loan", value=2)
        delayed = st.number_input("Num_of_Delayed_Payment", value=1)
        changed = st.number_input("Num_Credit_Inquiries", value=2)
        debt = st.number_input("Outstanding_Debt", value=1000.0)
        utilization = st.slider("Credit_Utilization_Ratio", 0.0, 100.0, 30.0)
        mix = st.selectbox("Credit_Mix", options=["Good", "Standard", "Bad"])
    
    submit = st.form_submit_button("Analizar Riesgo")

# 3. Procesamiento al presionar el botón
if submit:
    try:
        # CREACIÓN DEL DATAFRAME CON LAS 39 COLUMNAS
        # Se inicializa todo en 0.0
        input_df = pd.DataFrame(np.zeros((1, len(all_features))), columns=all_features)
        
        # ASIGNACIÓN DE VALORES (Basado en tu lista de 39 columnas)
        input_df.at[0, 'Age'] = age
        input_df.at[0, 'Annual_Income'] = income
        input_df.at[0, 'Monthly_Inhand_Salary'] = monthly
        input_df.at[0, 'Num_Bank_Accounts'] = accounts
        input_df.at[0, 'Num_Credit_Card'] = cards
        input_df.at[0, 'Interest_Rate'] = interest
        input_df.at[0, 'Num_of_Loan'] = loans
        input_df.at[0, 'Num_of_Delayed_Payment'] = delayed
        input_df.at[0, 'Num_Credit_Inquiries'] = changed
        input_df.at[0, 'Outstanding_Debt'] = debt
        input_df.at[0, 'Credit_Utilization_Ratio'] = utilization
        
        # Mapeo de Credit_Mix (Columna 12 en tu lista)
        mix_numeric = {'Bad': 0, 'Standard': 1, 'Good': 2}[mix]
        if 'Credit_Mix' in input_df.columns:
            input_df.at[0, 'Credit_Mix'] = mix_numeric

        # TRANSFORMACIÓN EN CASCADA
        # 1. Escalador (espera 39 entradas)
        X_scaled = scaler.transform(input_df)
        
        # 2. Selector (toma las 39 y elige las 25 o 10 que configuraste)
        X_kbest = selector.transform(X_scaled)
        
        # 3. PCA
        X_pca = pca.transform(X_kbest)
        
        # 4. Predicción con la Red Neuronal
        prediction = model.predict(X_pca)
        res = np.argmax(prediction)
        prob = np.max(prediction) * 100

        # RESULTADOS
        labels = {0: "POBRE 🔴", 1: "ESTÁNDAR 🟡", 2: "BUENO 🟢"}
        st.markdown("---")
        st.subheader(f"Resultado: {labels[res]}")
        st.write(f"Confianza del modelo: {prob:.2f}%")
        
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        st.write("Asegúrate de que los archivos .bin y .h5 estén actualizados en GitHub.")