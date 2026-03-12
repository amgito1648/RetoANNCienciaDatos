import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. Configuración de la página
st.set_page_config(page_title="Predictor de Crédito AI", page_icon="💳")

@st.cache_resource
def load_assets():
    nn_model = load_model('modelo_credito.h5')
    sc = joblib.load('scaler.bin')
    sel = joblib.load('selector.bin')
    pc = joblib.load('pca.bin')
    feat_names = joblib.load('todas_las_features.bin')
    return nn_model, sc, sel, pc, feat_names

model, scaler, selector, pca, all_features = load_assets()

st.title("💳 Sistema de Clasificación de Riesgo")

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

if submit:
    try:
        # CREACIÓN DEL DATAFRAME BLINDADO
        # Creamos un DF con ceros usando las columnas exactas que guardó el entrenamiento
        input_df = pd.DataFrame(np.zeros((1, len(all_features))), columns=all_features)
        
        # Diccionario con los valores capturados
        data_input = {
            'Age': age,
            'Annual_Income': income,
            'Monthly_Inhand_Salary': monthly,
            'Num_Bank_Accounts': accounts,
            'Num_Credit_Card': cards,
            'Interest_Rate': interest,
            'Num_of_Loan': loans,
            'Num_of_Delayed_Payment': delayed,
            'Num_Credit_Inquiries': changed,
            'Outstanding_Debt': debt,
            'Credit_Utilization_Ratio': utilization
        }

        # Llenar solo si la columna existe en el modelo entrenado
        for col, val in data_input.items():
            if col in input_df.columns:
                input_df.at[0, col] = val

        # MANEJO DE CREDIT_MIX (Numérico o Dummies)
        mix_numeric = {'Bad': 0, 'Standard': 1, 'Good': 2}[mix]
        if 'Credit_Mix' in input_df.columns:
            input_df.at[0, 'Credit_Mix'] = mix_numeric
        
        # Si el modelo espera dummies tipo 'Credit_Mix_Good'
        dummy_col = f"Credit_Mix_{mix}"
        if dummy_col in input_df.columns:
            input_df.at[0, dummy_col] = 1

        # TRANSFORMACIONES
        X_scaled = scaler.transform(input_df)
        X_kbest = selector.transform(X_scaled)
        X_pca = pca.transform(X_kbest)
        
        # PREDICCIÓN
        prediction = model.predict(X_pca)
        res = np.argmax(prediction)
        prob = np.max(prediction) * 100

        labels = {0: "POBRE 🔴", 1: "ESTÁNDAR 🟡", 2: "BUENO 🟢"}
        st.markdown("---")
        st.subheader(f"Resultado: {labels[res]}")
        st.write(f"Confianza: {prob:.2f}%")

    except Exception as e:
        st.error(f"Error detectado: {e}")
        # Esto te ayudará a ver qué columnas tiene tu modelo realmente
        st.write("Columnas que espera el modelo:", all_features)