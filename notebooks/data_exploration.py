# 1  Exploración simple del dataset
# ===============================

# Para luego llamarlas en otros scripts, se usa esto; 
# # Tu nuevo script
# import os
# from exploracion-datos  import count_images, load_and_show_examples

# # Definir rutas
# DATA_DIR = 'C:/Juanjo/Veolia/project3_claud/data/raw/'
# normal_dir = os.path.join(DATA_DIR, 'normal')
# anomaly_dir = os.path.join(DATA_DIR, 'anomaly')

# # Usar la función para contar imágenes
# normal_count, normal_sizes = count_images(normal_dir)
# anomaly_count, anomaly_sizes = count_images(anomaly_dir)

# # Ver ejemplos de imágenes
# normal_examples, anomaly_examples = load_and_show_examples(normal_dir, anomaly_dir)

# # Ahora puedes usar normal_examples y anomaly_examples para procesamiento adicional

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from collections import Counter

# Definir rutas (ajusta estas rutas según tu estructura de directorios)
DATA_DIR = 'data/raw/'

normal_dir = os.path.join(DATA_DIR, 'normal')
anomaly_dir = os.path.join(DATA_DIR, 'anomaly')

# Función para contar imágenes en un directorio
def count_images(directory):
    """Cuenta las imágenes en un directorio y calcula estadísticas básicas de tamaño."""
    if not os.path.exists(directory):
        print(f"El directorio {directory} no existe.")
        return 0, []
    
    # Contar todas las imágenes con diferentes extensiones
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
    
    num_images = len(image_paths)
    
    # Recopilar información de tamaño si hay imágenes
    image_sizes = []
    if num_images > 0:
        # Muestrear hasta 10 imágenes para calcular tamaños
        sample_paths = image_paths[:10] if num_images > 10 else image_paths
        
        for path in sample_paths:
            try:
                img = cv2.imread(path)
                if img is not None:
                    height, width = img.shape[:2]
                    image_sizes.append((height, width))
            except Exception as e:
                print(f"Error al leer {path}: {e}")
    
    return num_images, image_sizes

# Contar imágenes y obtener estadísticas
print("Analizando dataset...\n")

normal_count, normal_sizes = count_images(normal_dir)
anomaly_count, anomaly_sizes = count_images(anomaly_dir)

# Mostrar recuento de imágenes
print(f"Total de imágenes: {normal_count + anomaly_count}")
print(f"- Imágenes normales: {normal_count}")
print(f"- Imágenes con anomalías: {anomaly_count}")

# Mostrar estadísticas de tamaño
print("\nEstadísticas de tamaño de imágenes:")

if normal_sizes:
    heights, widths = zip(*normal_sizes)
    print(f"- Normal: media {np.mean(heights):.0f}x{np.mean(widths):.0f} píxeles")
    if len(set(heights)) > 1 or len(set(widths)) > 1:
        print(f"  Tamaños variables: {Counter(normal_sizes).most_common(3)}")
    else:
        print(f"  Tamaño único: {heights[0]}x{widths[0]} píxeles")

if anomaly_sizes:
    heights, widths = zip(*anomaly_sizes)
    print(f"- Anomalía: media {np.mean(heights):.0f}x{np.mean(widths):.0f} píxeles")
    if len(set(heights)) > 1 or len(set(widths)) > 1:
        print(f"  Tamaños variables: {Counter(anomaly_sizes).most_common(3)}")
    else:
        print(f"  Tamaño único: {heights[0]}x{widths[0]} píxeles")

# Función para cargar y mostrar imágenes de ejemplo
def load_and_show_examples(normal_dir, anomaly_dir, num_examples=2):
    """Carga y muestra ejemplos de imágenes normales y con anomalías."""
    
    # Función para cargar imágenes de un directorio
    def load_examples(directory, limit):
        images = []
        paths = []
        
        if not os.path.exists(directory):
            return images, paths
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        all_paths = []
        
        for ext in extensions:
            all_paths.extend(glob.glob(os.path.join(directory, ext)))
        
        # Tomar las primeras 'limit' imágenes
        sample_paths = all_paths[:limit] if all_paths else []
        
        for path in sample_paths:
            try:
                img = cv2.imread(path)
                if img is not None:
                    # Convertir de BGR a RGB para matplotlib
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
                    paths.append(path)
            except Exception as e:
                print(f"Error al cargar {path}: {e}")
        
        return images, paths
    
    # Cargar ejemplos
    normal_images, normal_paths = load_examples(normal_dir, num_examples)
    anomaly_images, anomaly_paths = load_examples(anomaly_dir, num_examples)
    
    # Configurar la visualización
    fig = plt.figure(figsize=(15, 10))
    
    # Mostrar imágenes normales
    for i, (img, path) in enumerate(zip(normal_images, normal_paths)):
        ax = fig.add_subplot(2, num_examples, i + 1)
        ax.imshow(img)
        ax.set_title(f"Normal")
        ax.axis('off')
    
    # Mostrar imágenes con anomalías
    for i, (img, path) in enumerate(zip(anomaly_images, anomaly_paths)):
        ax = fig.add_subplot(2, num_examples, num_examples + i + 1)
        ax.imshow(img)
        ax.set_title(f"Anomalía")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return normal_images, anomaly_images

# Mostrar ejemplos de imágenes
print("\nCargando imágenes de ejemplo...")
normal_examples, anomaly_examples = load_and_show_examples(normal_dir, anomaly_dir)

# Mostrar información sobre balanceo de clases
total = normal_count + anomaly_count
if total > 0:
    normal_percent = (normal_count / total) * 100
    anomaly_percent = (anomaly_count / total) * 100
    
    print(f"\nDistribución de clases:")
    print(f"- Normal: {normal_percent:.1f}%")
    print(f"- Anomalía: {anomaly_percent:.1f}%")
    
    if min(normal_count, anomaly_count) / max(normal_count, anomaly_count) < 0.5:
        print("⚠️ NOTA: El dataset está desbalanceado. Considera técnicas como class weights durante el entrenamiento.")
        