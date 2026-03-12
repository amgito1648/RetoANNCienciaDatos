import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. Configuración de la interfaz
st.set_page_config(page_title="Predictor de Crédito AI", page_icon="💳")

@st.cache_resource
def load_assets():
    
    model = load_model('modelo_credito.h5')
    scaler = joblib.load('scaler.bin')
    selector = joblib.load('selector.bin')
    pca = joblib.load('pca_v2.bin')  
    features = joblib.load('todas_las_features.bin')
    return model, scaler, selector, pca, features

# Carga de seguridad
try:
    model, scaler, selector, pca, all_features = load_assets()
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.info("Revisa que los archivos .bin y .h5 estén en la raíz de tu GitHub.")
    st.stop()

st.title("💳 Sistema de Clasificación de Riesgo")
st.write("Complete los datos para evaluar al cliente.")

# 2. Formulario
with st.form("main_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Edad", value=30)
        income = st.number_input("Ingreso Anual", value=50000.0)
        monthly = st.number_input("Salario Mensual", value=4000.0)
        accounts = st.number_input("Cuentas Bancarias", value=2)
        cards = st.number_input("Tarjetas Crédito", value=3)
        interest = st.number_input("Tasa Interés", value=12.0)
    with col2:
        loans = st.number_input("Préstamos", value=2)
        delayed = st.number_input("Pagos Atrasados", value=1)
        inquiries = st.number_input("Consultas Crédito", value=2)
        debt = st.number_input("Deuda Pendiente", value=1000.0)
        utilization = st.slider("Utilización Crédito", 0.0, 100.0, 30.0)
        mix = st.selectbox("Mix de Crédito", options=["Good", "Standard", "Bad"])
    
    submit = st.form_submit_button("Analizar Riesgo")


if submit:
    try:
        
        input_df = pd.DataFrame(np.zeros((1, len(all_features))), columns=all_features)
        
        # B. Mapeo de datos manuales
        mapping = {
            'Age': age, 'Annual_Income': income, 'Monthly_Inhand_Salary': monthly,
            'Num_Bank_Accounts': accounts, 'Num_Credit_Card': cards, 'Interest_Rate': interest,
            'Num_of_Loan': loans, 'Num_of_Delayed_Payment': delayed, 
            'Num_Credit_Inquiries': inquiries, 'Outstanding_Debt': debt, 
            'Credit_Utilization_Ratio': utilization
        }
        for col, val in mapping.items():
            if col in input_df.columns:
                input_df.at[0, col] = val

        # C. Mapeo de Credit_Mix
        mix_val = {'Bad': 0, 'Standard': 1, 'Good': 2}[mix]
        if 'Credit_Mix' in input_df.columns:
            input_df.at[0, 'Credit_Mix'] = mix_val

        
        input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Paso 2: Escalar
        X_scaled = scaler.transform(input_df)
        
       
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

       
        X_kbest = selector.transform(X_scaled)
        X_pca = pca.transform(X_kbest)
        
        # Paso 5: Predicción final
        prediction = model.predict(X_pca)
        clase = np.argmax(prediction)
        prob = np.max(prediction) * 100

        # E. Mostrar resultados
        etiquetas = {0: "POBRE 🔴", 1: "ESTÁNDAR 🟡", 2: "BUENO 🟢"}
        st.markdown("---")
        st.subheader(f"Resultado: {etiquetas[clase]}")
        st.write(f"Confianza: {prob:.2f}%")

    except Exception as e:
        st.error(f"Error técnico: {e}")
        st.write("Tip: Si el error persiste, asegúrate de haber actualizado el archivo PCA en GitHub.")