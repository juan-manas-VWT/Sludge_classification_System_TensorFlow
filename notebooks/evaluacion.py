# 4. Evaluación y Prueba del Modelo
# =============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, f1_score
import seaborn as sns
import cv2
import glob
import json
from tqdm.notebook import tqdm
import itertools

# # Configuración inicial
# plt.style.use('seaborn-whitegrid')
# np.random.seed(42)
# tf.random.set_seed(42)

# Definir rutas
# Definir rutas
VALIDATION_DIR = 'data/validation/'
TEST_DIR = 'data/test/'  # Nuevo
FINE_TUNED_DIR = 'models/fine_tuned/'
RESULTS_DIR = 'models/evaluation_results/'

# Crear directorio para resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

# 4.1 Cargar el modelo entrenado
# ---------------------------------------------

def load_trained_model(model_path):
    """
    Carga el modelo entrenado desde un archivo .h5 o directorio SavedModel.
    
    Args:
        model_path: Ruta al modelo guardado
    
    Returns:
        Modelo cargado
    """
    try:
        model = load_model(model_path)
        print(f"Modelo cargado desde {model_path}")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

# Cargar el mejor modelo
model_path = os.path.join(FINE_TUNED_DIR, 'best_model.h5')
model = load_trained_model(model_path)

if model is None:
    print("Intentando cargar el modelo final...")
    model_path = os.path.join(FINE_TUNED_DIR, 'final_model.h5')
    model = load_trained_model(model_path)
    
if model is None:
    print("Intentando cargar el modelo guardado...")
    model_path = os.path.join(FINE_TUNED_DIR, 'saved_model')
    model = load_trained_model(model_path)
    
if model is None:
    print("No se pudo cargar el modelo. Verifique las rutas y la existencia de los archivos.")
    import sys
    sys.exit(1)

# Cargar índices de clase
class_indices_path = os.path.join(FINE_TUNED_DIR, 'class_indices.json')
if os.path.exists(class_indices_path):
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    # Invertir el diccionario para obtener etiquetas a partir de índices
    idx_to_class = {v: k for k, v in class_indices.items()}
else:
    print("Archivo de índices de clase no encontrado. Usando valores predeterminados.")
    class_indices = {'normal':1, 'anomaly': 0}
    idx_to_class = {1: 'normal', 0: 'anomaly'}

print("Índices de clase:", class_indices)

# Cargar el umbral óptimo si está disponible
best_threshold_path = os.path.join(FINE_TUNED_DIR, 'best_threshold.txt')
if os.path.exists(best_threshold_path):
    with open(best_threshold_path, 'r') as f:
        best_threshold = float(f.read().strip())
    print(f"Usando umbral óptimo cargado: {best_threshold}")
else:
    best_threshold = 0.5
    print(f"Usando umbral predeterminado: {best_threshold}")

# 4.2 Función para predecir con umbral personalizado
# ---------------------------------------------

def predict_with_custom_threshold(model, image_path, threshold=best_threshold):
    """
    Predice la clase de una imagen con un umbral personalizado.
    
    Args:
        model: Modelo entrenado
        image_path: Ruta a la imagen
        threshold: Umbral de decisión
    
    Returns:
        Clase predicha y probabilidad
    """
    # Cargar y preprocesar imagen
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640,448))  # Ajustar al tamaño de entrada del modelo
    
    # Normalizar
    img = img.astype(np.float32) / 255.0
    
    # Ampliar dimensiones para batch
    img_array = np.expand_dims(img, axis=0)
    
    # Predecir
    prediction = model.predict(img_array)[0][0]
    
    # Aplicar umbral personalizado
    class_name = "anomaly" if prediction <= threshold else "normal"
    
    return class_name, prediction

# Probar la función con una imagen de ejemplo
example_anomaly_path = glob.glob(os.path.join(TEST_DIR, 'anomaly', '*'))[0]
class_name, prediction = predict_with_custom_threshold(model, example_anomaly_path, threshold=best_threshold)
print(f"Ejemplo - Clase: {class_name}, Probabilidad: {prediction:.4f}")

# 4.3 Preparar datos de validación
# ---------------------------------------------

# Determinar tamaño de imagen basado en la entrada del modelo
input_shape = model.input_shape[1:3]
print(f"Tamaño de entrada del modelo: {input_shape}")

# Generador para datos de prueba (en lugar de validación)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,  # Usar TEST_DIR en lugar de VALIDATION_DIR
    target_size=input_shape,
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# 4.4 Evaluación del modelo
# ---------------------------------------------

# Evaluar el modelo en el conjunto de prueba
print("Evaluando modelo en conjunto de prueba...")
evaluation = model.evaluate(test_generator, verbose=1)

# Mostrar resultados de la evaluación
metrics = model.metrics_names
evaluation_results = dict(zip(metrics, evaluation))
print("\nResultados de la evaluación:")
for metric, value in evaluation_results.items():
    print(f"{metric}: {value:.4f}")

# 4.5 Predicciones y métricas detalladas
# ---------------------------------------------

# Obtener predicciones para todas las imágenes de validación
print("Generando predicciones para todas las imágenes...")

# Restablecer el generador
test_generator.reset()

# Número total de muestras
n_samples = test_generator.samples

# Arreglos para almacenar predicciones y etiquetas reales
y_true = np.zeros(n_samples, dtype=int)
y_pred = np.zeros(n_samples)
y_pred_binary = np.zeros(n_samples, dtype=int)

# Procesar todas las muestras
for i in tqdm(range(n_samples)):
    # Obtener una muestra
    x, y = next(test_generator)
    # Almacenar etiqueta real
    y_true[i] = int(y[0])
    # Generar predicción
    pred = model.predict(x, verbose=0)[0][0]
    y_pred[i] = pred
    y_pred_binary[i] = 1 if pred >= best_threshold else 0

# 4.6 Visualización de resultados
# ---------------------------------------------

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred_binary)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[idx_to_class[0], idx_to_class[1]],
            yticklabels=[idx_to_class[0], idx_to_class[1]])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()

# Informe de clasificación
print("\nInforme de Clasificación:")
report = classification_report(y_true, y_pred_binary, 
                              target_names=[idx_to_class[0], idx_to_class[1]])
print(report)

# Guardar reporte como archivo de texto
with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.show()

# Curva Precision-Recall
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.axhline(y=sum(y_true)/len(y_true), color='r', linestyle='--', 
           label=f'Clase positiva: {sum(y_true)/len(y_true):.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend(loc="lower left")
plt.savefig(os.path.join(RESULTS_DIR, 'pr_curve.png'), dpi=300, bbox_inches='tight')
plt.show()

# Distribución de probabilidades de predicción
plt.figure(figsize=(10, 6))
sns.histplot(y_pred[y_true==0], color='red', alpha=0.5, bins=30, label='Anomalía')
sns.histplot(y_pred[y_true==1], color='green', alpha=0.5, bins=30, label='Normal')
plt.axvline(x=best_threshold, color='black', linestyle='--', label=f'Umbral ({best_threshold:.2f})')
plt.xlabel('Probabilidad de Anomalía')
plt.ylabel('Frecuencia')
plt.title('Distribución de Probabilidades de Predicción')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4.7 Visualizar predicciones en imágenes individuales
# ---------------------------------------------

def visualize_predictions(model, validation_dir, class_indices, n_samples=5, threshold=best_threshold):
    """
    Visualiza predicciones en imágenes individuales.
    
    Args:
        model: Modelo entrenado
        validation_dir: Directorio con imágenes de validación
        class_indices: Diccionario con índices de clase
        n_samples: Número de muestras a visualizar por clase
        threshold: Umbral de clasificación personalizado
    """
    # Invertir diccionario de índices
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Obtener rutas de imágenes
    normal_paths = glob.glob(os.path.join(validation_dir, 'normal', '*'))[:n_samples]
    anomaly_paths = glob.glob(os.path.join(validation_dir, 'anomaly', '*'))[:n_samples]
    
    image_paths = normal_paths + anomaly_paths
    true_labels = ['normal'] * len(normal_paths) + ['anomaly'] * len(anomaly_paths)
    
    # Preparar figura
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))
    
    # Procesar cada imagen
    for i, (img_path, true_label) in enumerate(zip(image_paths, true_labels)):
        # Determinar índice de subplot
        row = 0 if true_label == 'normal' else 1
        col = i if row == 0 else i - n_samples
        
        # Cargar y preprocesar imagen
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, model.input_shape[1:3])
        img_array = img_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Generar predicción
        pred = model.predict(img_array, verbose=0)[0][0]
        pred_label = 'anomaly' if pred <= threshold else 'normal'
        
        # Determinar color (verde para correcto, rojo para incorrecto)
        color = 'green' if pred_label == true_label else 'red'
        
        # Mostrar imagen
        axes[row, col].imshow(img_resized)
        axes[row, col].set_title(f"Real: {true_label}\nPred: {pred_label} ({pred:.2f})", 
                                 color=color)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
    plt.show()

# Visualizar algunas predicciones
visualize_predictions(model, VALIDATION_DIR, class_indices, n_samples=5, threshold=best_threshold)

# 4.8 Análisis de umbrales
# ---------------------------------------------

def plot_threshold_metrics(y_true, y_pred):
    """
    Analiza el efecto de diferentes umbrales en las métricas de rendimiento.
    
    Args:
        y_true: Etiquetas reales
        y_pred: Probabilidades predichas
    """
    thresholds = np.linspace(0, 1, 100)
    accuracy = []
    precision = []
    recall = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calcular métricas
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        # Accuracy
        current_accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy.append(current_accuracy)
        
        # Precision
        current_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision.append(current_precision)
        
        # Recall
        current_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall.append(current_recall)
        
        # F1 Score
        current_f1 = 2 * (current_precision * current_recall) / (current_precision + current_recall) \
            if (current_precision + current_recall) > 0 else 0
        f1_scores.append(current_f1)
    
    # Encontrar el mejor umbral para F1 score
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresholds[best_threshold_idx]
    
    # Visualizar métricas según umbral
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracy, label='Accuracy')
    plt.plot(thresholds, precision, label='Precision')
    plt.plot(thresholds, recall, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.axvline(x=best_threshold_f1, color='black', linestyle='--', 
                label=f'Mejor umbral F1: {best_threshold_f1:.2f}')
    if best_threshold != 0.5:
        plt.axvline(x=best_threshold, color='red', linestyle=':', 
                    label=f'Umbral cargado: {best_threshold:.2f}')
    plt.xlabel('Umbral')
    plt.ylabel('Valor de métrica')
    plt.title('Métricas vs. Umbral de Decisión')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_threshold_f1

# Analizar y encontrar el mejor umbral
best_threshold_f1 = plot_threshold_metrics(y_true, y_pred)
print(f"Mejor umbral F1 encontrado: {best_threshold_f1:.4f}")
print(f"Umbral utilizado en evaluación: {best_threshold:.4f}")

# 4.9 Visualizar casos difíciles (falsos positivos y falsos negativos)
# ---------------------------------------------

# Identificar falsos positivos y falsos negativos
false_positives = np.where((y_pred_binary == 1) & (y_true == 0))[0]
false_negatives = np.where((y_pred_binary == 0) & (y_true == 1))[0]

print(f"Número de falsos positivos: {len(false_positives)}")
print(f"Número de falsos negativos: {len(false_negatives)}")

# Función para visualizar casos difíciles
def show_misclassified(generator, indices, y_pred, idx_to_class, title, threshold=best_threshold):
    if len(indices) == 0:
        print(f"No hay {title} para mostrar")
        return
    
    # Restablecer generador
    generator.reset()
    
    # Almacenar información de muestras
    samples = []
    
    for idx in indices:
        # Avanzar en el generador hasta la muestra deseada
        generator.reset()
        for i in range(idx+1):
            x, y = next(generator)
        
        # Almacenar información
        samples.append({
            'image': x[0],
            'true_label': idx_to_class[int(y[0])],
            'pred_score': y_pred[idx],
            'pred_label': idx_to_class[1] if y_pred[idx] >= threshold else idx_to_class[0]
        })
    
    # Visualizar muestras
    n_cols = min(5, len(samples))
    n_rows = (len(samples) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, sample in enumerate(samples[:n_rows*n_cols]):
        axes[i].imshow(sample['image'])
        axes[i].set_title(f"Real: {sample['true_label']}\nPred: {sample['pred_label']}\nScore: {sample['pred_score']:.2f}")
        axes[i].axis('off')
    
    # Ocultar ejes vacíos
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{title.lower().replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.show()

# Visualizar falsos positivos (máximo 10)
fp_indices = false_positives[:10]
if len(fp_indices) > 0:
    show_misclassified(test_generator, fp_indices, y_pred, idx_to_class, "Falsos Positivos")

# Visualizar falsos negativos (máximo 10)
fn_indices = false_negatives[:10]
if len(fn_indices) > 0:
    show_misclassified(test_generator, fn_indices, y_pred, idx_to_class, "Falsos Negativos")

# 4.10 Resumen de la evaluación
# ---------------------------------------------

# Crear un resumen consolidado de la evaluación
evaluation_summary = {
    'accuracy': evaluation_results.get('accuracy', 0),
    'precision': evaluation_results.get('precision', 0) if 'precision' in evaluation_results else 
                  evaluation_results.get('precision_1', 0),
    'recall': evaluation_results.get('recall', 0) if 'recall' in evaluation_results else 
              evaluation_results.get('recall_1', 0),
    'auc': evaluation_results.get('auc', 0) if 'auc' in evaluation_results else 
           evaluation_results.get('auc_1', 0),
    'best_threshold': best_threshold,
    'best_threshold_f1': best_threshold_f1,
    'roc_auc': roc_auc,
    'pr_auc': pr_auc,
    'confusion_matrix': cm.tolist(),
    'false_positives': len(false_positives),
    'false_negatives': len(false_negatives)
}

# Guardar resumen como JSON
with open(os.path.join(RESULTS_DIR, 'evaluation_summary.json'), 'w') as f:
    json.dump(evaluation_summary, f, indent=4)

print(f"Resumen de evaluación guardado en {os.path.join(RESULTS_DIR, 'evaluation_summary.json')}")

# Calcular métricas con el mejor umbral F1
if best_threshold_f1 != best_threshold:
    y_pred_best_f1 = (y_pred >= best_threshold_f1).astype(int)
    cm_best_f1 = confusion_matrix(y_true, y_pred_best_f1)

    print(f"\nMatriz de confusión con el mejor umbral F1 ({best_threshold_f1:.4f}):")
    print(cm_best_f1)

    print("\nInforme de clasificación con el mejor umbral F1:")
    print(classification_report(y_true, y_pred_best_f1, 
                           target_names=[idx_to_class[0], idx_to_class[1]]))

print("Evaluación completada con éxito!")