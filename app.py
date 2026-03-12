import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. Configuración de la página
st.set_page_config(page_title="Predictor de Crédito AI", page_icon="💳")

@st.cache_resource
def load_assets():
    # Carga de archivos exportados desde tu notebook
    nn_model = load_model('modelo_credito.h5')
    sc = joblib.load('scaler.bin')
    sel = joblib.load('selector.bin')
    pc = joblib.load('pca.bin')
    feat_names = joblib.load('todas_las_features.bin')
    return nn_model, sc, sel, pc, feat_names

model, scaler, selector, pca, all_features = load_assets()

st.title("💳 Sistema de Clasificación de Riesgo")
st.write("Introduce los datos del cliente para evaluar su perfil crediticio.")

# 2. Formulario con los nombres exactos de tu dataset
with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (Edad)", value=30)
        income = st.number_input("Annual_Income (Ingreso Anual)", value=50000.0)
        monthly = st.number_input("Monthly_Inhand_Salary (Salario Mensual)", value=4000.0)
        accounts = st.number_input("Num_Bank_Accounts", value=2)
        cards = st.number_input("Num_Credit_Card", value=3)
        interest = st.number_input("Interest_Rate", value=12.0)

    with col2:
        loans = st.number_input("Num_of_Loan", value=2)
        delayed = st.number_input("Num_of_Delayed_Payment", value=1)
        changed = st.number_input("Num_Credit_Inquiries", value=2)
        debt = st.number_input("Outstanding_Debt (Deuda Pendiente)", value=1000.0)
        utilization = st.slider("Credit_Utilization_Ratio", 0.0, 100.0, 30.0)
        mix = st.selectbox("Credit_Mix", options=["Good", "Standard", "Bad"])

    submit = st.form_submit_button("Analizar Riesgo")

if submit:
    # ... (dentro del bloque if submit:)

# 1. Crear el DataFrame con TODAS las columnas
input_df = pd.DataFrame(0, index=[0], columns=all_features)

# 2. Llenar los datos (Asegúrate de que los nombres coincidan con tu CSV)
mapping = {
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
    'Credit_Utilization_Ratio': utilization,
    'Credit_Mix': {'Bad': 0, 'Standard': 1, 'Good': 2}[mix]
}

for col, val in mapping.items():
    if col in input_df.columns:
        input_df[col] = val

# --- EL TRUCO PARA EL ERROR ESTÁ AQUÍ ---
# Reordenamos las columnas para que estén en el mismo orden que 'all_features'
input_df = input_df[all_features] 

# 3. Transformaciones en cadena
X_scaled = scaler.transform(input_df)
X_kbest = selector.transform(X_scaled)  # Aquí es donde fallaba, ahora recibirá el orden correcto
X_pca = pca.transform(X_kbest)

# 4. Predicción
prediction = model.predict(X_pca)
# ... el resto del código sigue igual
    # A. Crear DataFrame base con todas las columnas en 0
    input_df = pd.DataFrame(0, index=[0], columns=all_features)
    
    # B. Mapeo manual según tu preprocesamiento
    mapping = {
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
        'Credit_Utilization_Ratio': utilization,
        'Credit_Mix': {'Bad': 0, 'Standard': 1, 'Good': 2}[mix]
    }
    
    for col, val in mapping.items():
        if col in input_df.columns:
            input_df[col] = val

    # C. Pipeline de transformación
    # 1. Escalar -> 2. SelectKBest (10 mejores) -> 3. PCA -> 4. Predicción
    X_scaled = scaler.transform(input_df)
    X_kbest = selector.transform(X_scaled)
    X_pca = pca.transform(X_kbest)
    
    prediction = model.predict(X_pca)
    res = np.argmax(prediction)
    prob = np.max(prediction) * 100

    # D. Mostrar resultado
    labels = {0: "POBRE 🔴", 1: "ESTÁNDAR 🟡", 2: "BUENO 🟢"}
    st.subheader(f"Resultado: {labels[res]}")
    st.write(f"Confianza del modelo: {prob:.2f}%")