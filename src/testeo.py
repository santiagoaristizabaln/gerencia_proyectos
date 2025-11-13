import numpy as np
import os
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# --- RUTAS Y CARGA DE DATOS ---
CARPETA_PROCESSED = 'data/processed'
CARPETA_MODELOS = 'models'

print("--- Carga de Datos Procesados y Conversión a Arreglo Denso ---")
try:
    # 1. Cargar los datos procesados desde archivos .npy
    # Usamos allow_pickle=True para que NumPy pueda manejar objetos CSR Matrix
    X_train_processed = np.load(os.path.join(CARPETA_PROCESSED, 'X_train_processed.npy'), allow_pickle=True).item()
    X_test_processed = np.load(os.path.join(CARPETA_PROCESSED, 'X_test_processed.npy'), allow_pickle=True).item()
    y_train = np.load(os.path.join(CARPETA_PROCESSED, 'y_train.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(CARPETA_PROCESSED, 'y_test.npy'), allow_pickle=True)
    
    # 2. CORRECCIÓN CLAVE: Convertir explícitamente a formato denso (.toarray())
    # NOTA: Usamos .item() en la carga porque a veces np.save guarda la matriz dispersa 
    # dentro de un array de NumPy de un solo elemento (lo cual requiere .item() para acceder al CSR Matrix).
    X_train_processed = X_train_processed.toarray()
    X_test_processed = X_test_processed.toarray()
    
    print("Convertido X_train y X_test de matriz dispersa a arreglo denso.")
    print("✅ Datos cargados y preparados exitosamente.")
except FileNotFoundError:
    print(f"❌ ERROR: Archivos .npy no encontrados en {CARPETA_PROCESSED}.")
    exit()
except AttributeError:
     # Captura si el .toarray() falla, indicando que el objeto no es una matriz dispersa válida.
     print("⚠️ Aviso: El objeto cargado no es una matriz dispersa. Intentando usarlo directamente...")
except Exception as e:
    print(f"❌ Ocurrió un error inesperado durante la carga/conversión: {e}")
    exit()

print("-" * 50)

## 📈 Entrenamiento y Testeo del Modelo

# --- 1. ENTRENAMIENTO DEL MODELO ---
print("--- 1. Entrenamiento del Modelo ---")
modelo = LogisticRegression(solver='liblinear', random_state=42)
modelo.fit(X_train_processed, y_train) 
print("Modelo de Regresión Logística entrenado exitosamente.")

# Guardar el modelo entrenado
RUTA_MODELO_GUARDADO = os.path.join(CARPETA_MODELOS, 'modelo_regresion_logistica.pkl')
joblib.dump(modelo, RUTA_MODELO_GUARDADO)
print(f"✅ Modelo guardado en: {RUTA_MODELO_GUARDADO}")
print("-" * 50)


# --- 2. GENERACIÓN DE PREDICCIONES Y EVALUACIÓN ---
print("--- 2. Generación de Predicciones y Evaluación ---")

y_pred = modelo.predict(X_test_processed)
y_prob = modelo.predict_proba(X_test_processed)[:, 1]

# Métricas principales
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

with open('metrics.json', 'w') as f:
    json.dump({'accuracy': accuracy}, f)
    json.dump({'roc_auc': roc_auc}, f)
    json.dump({'f1': f1}, f)

print(f"Precisión (Accuracy): {accuracy:.4f}")
print(f"AUC-ROC Score: {roc_auc:.4f}")
print(f"F1-Score: {f1:.4f}")
print("-" * 50)

# Reporte de Clasificación Detallado
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Default (0)', 'Default (1)']))

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión (Array de NumPy):")
print(conf_matrix)

# Visualización (requiere Matplotlib/Seaborn instalados)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default (0)', 'Default (1)'],
            yticklabels=['No Default (0)', 'Default (1)'])
plt.title('Matriz de Confusión (Regresión Logística)')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.show()

print("-" * 50)
print("✅ Testeo del modelo completado y evaluado.")
