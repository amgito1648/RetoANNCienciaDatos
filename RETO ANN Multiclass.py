import pandas as pd
import numpy as np

df = pd.read_csv('riesgo.xlsx - Sheet1.csv', sep=None, engine='python')
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
cols_to_drop = ['Customer_ID', 'Name', 'SSN']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

if 'Type_of_Loan' in df.columns:
    df['Type_of_Loan'] = df['Type_of_Loan'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

if 'Credit_Mix' in df.columns:
    df['Credit_Mix'] = df['Credit_Mix'].astype(str).str.strip().map({'Bad': 0, 'Standard': 1, 'Good': 2})
if 'Payment_of_Min_Amount' in df.columns:
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].astype(str).str.strip().map({'No': 0, 'Yes': 1})

cols_to_dummy = [c for c in ['Occupation', 'Payment_Behaviour'] if c in df.columns]
df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True)

df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.median())

# Asegurar Target
df = df.dropna(subset=['Credit_Score'])
df['Credit_Score'] = df['Credit_Score'].astype(int)

y = df['Credit_Score']
X = df.drop(columns=['Credit_Score'])

print(f"Dataset listo: {X.shape[1]} columnas originales.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Dividir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. SELECT K BEST (Las 10 mejores)
selector = SelectKBest(score_func=f_classif, k=10)
X_train_kbest = selector.fit_transform(np.nan_to_num(X_train_scaled), y_train)
X_test_kbest = selector.transform(np.nan_to_num(X_test_scaled))

# Ver cuáles quedaron para tu Streamlit
cols_idxs = selector.get_support(indices=True)
features_seleccionadas = X.columns[cols_idxs].tolist()

print(f"✅ Selección completada. Las 10 mejores son: {features_seleccionadas}")

from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers, models

# PCA sobre las 10 elegidas
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train_kbest)
X_test_pca = pca.transform(X_test_kbest)

# Red Neuronal Profunda
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

history = model.fit(X_train_pca, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1)


import joblib

model.save('modelo_credito.h5')
joblib.dump(scaler, 'scaler.bin')
joblib.dump(pca, 'pca.bin')
joblib.dump(selector, 'selector.bin') # ¡NUEVO!
joblib.dump(X.columns.tolist(), 'todas_las_features.bin') 

print("Archivos listos para subir a GitHub.")


import joblib
import os

# Crear la carpeta si no existe (por si acaso)
path = './' 

# 1. Guardar el modelo de Keras (.h5)
model.save(os.path.join(path, 'modelo_credito.h5'))

# 2. Guardar los transformadores de Scikit-Learn (.bin)
joblib.dump(scaler, os.path.join(path, 'scaler.bin'))
joblib.dump(selector, os.path.join(path, 'selector.bin'))
joblib.dump(pca, os.path.join(path, 'pca.bin'))

# 3. Guardar la lista de columnas originales
# Usamos X.columns porque son las que el modelo espera antes de procesar
joblib.dump(X.columns.tolist(), os.path.join(path, 'todas_las_features.bin'))

print("✅ ¡Archivos creados con éxito!")
print("Busca en tu carpeta: modelo_credito.h5, scaler.bin, selector.bin, pca.bin, todas_las_features.bin")