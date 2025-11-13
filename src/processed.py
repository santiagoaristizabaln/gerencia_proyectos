import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- RUTAS DE ARCHIVOS ---
RUTA_DATASET_LIMPIO = 'data/processed/UCI_Credit_Card_Limpio.csv'
CARPETA_GUARDADO = 'data/processed'

# --- CARGA DE DATOS ---
try:
    df = pd.read_csv(RUTA_DATASET_LIMPIO)
    print(f"✅ DataFrame limpio cargado desde: {RUTA_DATASET_LIMPIO}")
except FileNotFoundError:
    print(f"❌ ERROR: No se encontró el archivo limpio en {RUTA_DATASET_LIMPIO}.")
    exit()

print("\n--- Inicio del Procesamiento de Datos ---")
# -------------------------------------------------------------------------


# --- 1. DEFINICIÓN DE VARIABLES ---
TARGET_COLUMN = 'default_payment_next_month'

# Excluimos 'id'
df.drop('id', axis=1, inplace=True, errors='ignore')

# Separación de X (características) e y (objetivo)
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

print(f"Variable Objetivo (y): {TARGET_COLUMN}")
print("-" * 30)


# --- 2. IDENTIFICACIÓN DE COLUMNAS (Nombres Normalizados) ---
numerical_features = [
    'limit_bal', 'age', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4',
    'bill_amt5', 'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4',
    'pay_amt5', 'pay_amt6'
]
categorical_features = [
    'sex', 'education', 'marriage', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6'
]

print(f"Columnas Numéricas a Escalar: {len(numerical_features)}")
print(f"Columnas Categóricas a Codificar: {len(categorical_features)}")
print("-" * 30)


# --- 3. CREACIÓN DE PIPELINE DE TRANSFORMACIÓN ---
numerical_transformer = StandardScaler()
# El ColumnTransformer producirá una matriz dispersa (csr_matrix)
categorical_transformer = OneHotEncoder(handle_unknown='ignore') 

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

print("Pipeline de preprocesamiento creado.")
print("-" * 30)


# --- 4. DIVISIÓN DEL DATASET ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. APLICACIÓN DEL PIPELINE Y GUARDADO ---
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Guardar el objeto 'preprocessor' para usarlo en producción
joblib.dump(preprocessor, os.path.join(CARPETA_GUARDADO, 'preprocessor.pkl'))
print(f"✅ Objeto 'preprocessor' guardado.")

# Guardar los arreglos de NumPy
np.save(os.path.join(CARPETA_GUARDADO, 'X_train_processed.npy'), X_train_processed)
np.save(os.path.join(CARPETA_GUARDADO, 'X_test_processed.npy'), X_test_processed)
np.save(os.path.join(CARPETA_GUARDADO, 'y_train.npy'), y_train)
np.save(os.path.join(CARPETA_GUARDADO, 'y_test.npy'), y_test)

print("\n--- Guardado de Datos Procesados ---")
print(f"Archivos .npy guardados exitosamente en: {CARPETA_GUARDADO}")
print(f"X_train_processed shape final: {X_train_processed.shape}")
