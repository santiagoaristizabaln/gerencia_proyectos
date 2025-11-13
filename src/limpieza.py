import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- RUTAS DE ARCHIVOS ---
RUTA_DATASET_RAW = 'data/raw/UCI_Credit_Card.csv'
CARPETA_PROCESSED = 'data/processed'
RUTA_GUARDADO = os.path.join(CARPETA_PROCESSED, 'UCI_Credit_Card_Limpio.csv')

# --- CARGA DEL DATASET ---
try:
    df = pd.read_csv(RUTA_DATASET_RAW)
    print(f"✅ Archivo cargado exitosamente desde: {RUTA_DATASET_RAW}")
except FileNotFoundError:
    print(f"❌ ERROR: Archivo no encontrado en la ruta {RUTA_DATASET_RAW}.")
    exit()

# --- 1. NORMALIZACIÓN INICIAL Y EXPLORACIÓN ---
# 1.1 Normalización de nombres de columnas (minúsculas y guiones bajos)
df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]+', '_', regex=True)
print("Nombres de columnas normalizados.")

# 1.2 Información General
print("--- 1. Información General ---")
print(f"Dimensiones iniciales del DataFrame: {df.shape}")
df.info()
print("-" * 50)


# --- 2. MANEJO DE DUPLICADOS ---
print("--- 2. Datos Duplicados ---")
# Contamos y eliminamos duplicados (excluyendo la columna 'id')
filas_duplicadas = df.drop(columns=['id'], errors='ignore').duplicated().sum() 
print(f"Número de filas duplicadas encontradas (excluyendo ID): {filas_duplicadas}")

if filas_duplicadas > 0:
    df.drop_duplicates(subset=df.columns.drop('id', errors='ignore'), inplace=True)
    print(f"Se eliminaron {filas_duplicadas} filas duplicadas.")
print(f"Dimensiones después de eliminar duplicados: {df.shape}")
print("-" * 50)


# --- 3. CORRECCIÓN DE VALORES ANÓMALOS CATEÓRICOS (ESPECÍFICO DE UCI DATASET) ---
print("--- 3. Corrección de Valores Anómalos Categóricos ---")

# 3.1 Normalización de 'education' (0, 5, 6 -> 4 [Otros/Desconocido])
df['education'].replace({0: 4, 5: 4, 6: 4}, inplace=True)
print("Normalizado: 'education'.")

# 3.2 Normalización de 'marriage' (0 -> 3 [Otros/Desconocido])
df['marriage'].replace({0: 3}, inplace=True)
print("Normalizado: 'marriage'.")

print("-" * 50)


# --- 4. MANEJO DE VALORES FALTANTES (NaN / Null) ---
print("--- 4. Valores Faltantes ---")
valores_faltantes = df.isnull().sum()
print("Cantidad de NaN por columna:", valores_faltantes[valores_faltantes > 0])
print("-" * 50)


# --- 5. GUARDA DEL DATASET LIMPIO EN data/processed ---
# Aseguramos que la carpeta exista
Path(CARPETA_PROCESSED).mkdir(parents=True, exist_ok=True)

try:
    df.to_csv(RUTA_GUARDADO, index=False)
    print("--- ¡LIMPIEZA AVANZADA COMPLETADA Y GUARDADA! ---")
    print(f"✅ Archivo limpio guardado en: {RUTA_GUARDADO}")
    print(f"DataFrame final con {df.shape[0]} filas y {df.shape[1]} columnas.")
except Exception as e:
    print(f"❌ ERROR al intentar guardar el archivo: {e}")
