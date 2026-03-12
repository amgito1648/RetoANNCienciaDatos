import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers, models

# --- 1. CARGA ---
df = pd.read_csv('riesgo.xlsx - Sheet1.csv', sep=None, engine='python')
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
cols_to_drop = ['Customer_ID', 'Name', 'SSN', 'ID']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# --- 2. PREPROCESAMIENTO ---
if 'Type_of_Loan' in df.columns:
    df['Type_of_Loan'] = df['Type_of_Loan'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

if 'Credit_Mix' in df.columns:
    df['Credit_Mix'] = df['Credit_Mix'].astype(str).str.strip().map({'Bad': 0, 'Standard': 1, 'Good': 2})

if 'Payment_of_Min_Amount' in df.columns:
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].astype(str).str.strip().map({'No': 0, 'Yes': 1})

cols_to_dummy = [c for c in ['Occupation', 'Payment_Behaviour'] if c in df.columns]
df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True)

# --- 3. LIMPIEZA CRÍTICA (Aquí estaba el fallo) ---
df = df.apply(pd.to_numeric, errors='coerce')
# Rellenamos NaNs con la mediana
df = df.fillna(df.median())

# ELIMINAR COLUMNAS CONSTANTES (Si todos los valores son iguales, la columna no sirve)
df = df.loc[:, (df != df.iloc[0]).any()] 

# Asegurar Target
df = df.dropna(subset=['Credit_Score'])
df['Credit_Score'] = df['Credit_Score'].astype(int)

y = df['Credit_Score']
X = df.drop(columns=['Credit_Score'])

# --- 4. ENTRENAMIENTO ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
# Usamos np.nan_to_num por si acaso quedara algún residuo
X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train))
X_test_scaled = np.nan_to_num(scaler.transform(X_test))

# Select K Best
selector = SelectKBest(score_func=f_classif, k=min(10, X_train_scaled.shape[1]))
X_train_kbest = selector.fit_transform(X_train_scaled, y_train)
X_test_kbest = selector.transform(X_test_scaled)

# PCA
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train_kbest)
X_test_pca = pca.transform(X_test_kbest)

# --- 5. MODELO ---
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train_pca.shape[1],)),
    BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0003), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_pca, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1)

# --- 6. EXPORTACIÓN ---
model.save('modelo_credito.h5')
joblib.dump(scaler, 'scaler.bin')
joblib.dump(selector, 'selector.bin')
joblib.dump(pca, 'pca.bin')
joblib.dump(X.columns.tolist(), 'todas_las_features.bin')

print("✅ ¡LOGRADO! Archivos generados sin errores.")