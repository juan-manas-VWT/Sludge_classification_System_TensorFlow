# 3 training 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import datetime
import json
import cv2



# # Configuración inicial
# plt.style.use('seaborn-whitegrid')
# np.random.seed(42)
# tf.random.set_seed(42)

# Definir rutas - Aquí se usan directamente las carpetas de train/validation
TRAIN_DIR = 'data/train/'
VALIDATION_DIR = 'data/validation/'
MODELS_DIR = 'models/'
FINE_TUNED_DIR = os.path.join(MODELS_DIR, 'fine_tuned')

# Crear directorios si no existen
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FINE_TUNED_DIR, exist_ok=True)

                                            # # 3.1 Cargar configuración
                                            # # ---------------------------------------------
                                            # PROCESSED_DIR = 'data/processed/'
                                            # def load_data_config(config_path):
                                            #     """Carga la configuración de datos desde un archivo JSON."""
                                            #     with open(config_path, 'r') as f:
                                            #         return json.load(f)

                                            # # Cargar configuración
                                            # config_path = os.path.join(PROCESSED_DIR, 'data_config.json')
                                            # if os.path.exists(config_path):
                                            #     config = load_data_config(config_path)
                                            #     print("Configuración cargada desde:", config_path)
                                                
                                            #     # Extraer parámetros de configuración
                                            #     IMG_SIZE = tuple(config['image_size'])
                                            #     BATCH_SIZE = config['batch_size']
                                            #     steps_per_epoch = config['steps_per_epoch']
                                            #     validation_steps = config['validation_steps']
                                            #     class_indices = config['class_indices']
                                            #     print("Configuración cargada:")
                                            #     print(f"Tamaño de imagen: {IMG_SIZE}")
                                            #     print(f"Tamaño de lote: {BATCH_SIZE}")
                                            #     print(f"Pasos por época: {steps_per_epoch}")
                                            #     print(f"Índices de clase: {class_indices}")
                                            # else:
                                            #     print("Archivo de configuración no encontrado. Usando valores predeterminados.")
                                            #     IMG_SIZE = (224, 224)
                                            #     BATCH_SIZE = 32
                                            #     # Calcular parámetros basados en el directorio
                                            #     train_samples = len(os.listdir(os.path.join(TRAIN_DIR, 'normal'))) + \
                                            #                    len(os.listdir(os.path.join(TRAIN_DIR, 'anomaly')))
                                            #     validation_samples = len(os.listdir(os.path.join(VALIDATION_DIR, 'normal'))) + \
                                            #                         len(os.listdir(os.path.join(VALIDATION_DIR, 'anomaly')))
                                            #     steps_per_epoch = train_samples // BATCH_SIZE
                                            #     validation_steps = validation_samples // BATCH_SIZE
#                                                 class_indices = {'normal':1, 'anomaly': 0}

# New img size after cropping to follow the rectangular shape
IMG_SIZE = (186, 400)  # heith x width 
# IMG_SIZE = (372, 800)  # The double



# 3.2 Prepare data generators
# ---------------------------------------------
def preprocess_image(img):
    """
    Crops a specific region of the image and resizes it to the desired size.
    If the image is too small, the crop is adapted proportionally.
    """
    # Obtener dimensiones de la imagen
    img_height, img_width = img.shape[:2]
    
    # Coordenadas originales para el recorte (para imágenes de tamaño completo)
    target_start_x = 943
    target_start_y = 861
    target_width = 3569
    target_height = 1651
    
    # Calcular la proporción de la imagen actual respecto a un tamaño de referencia
    # (asumiendo que las coordenadas originales son para imágenes de 4512x2512)
    reference_width = 4512
    reference_height = 2512
    
    width_ratio = img_width / reference_width
    height_ratio = img_height / reference_height
    
    # Ajustar las coordenadas según la proporción
    start_x = int(target_start_x * width_ratio)
    start_y = int(target_start_y * height_ratio)
    width = int(target_width * width_ratio)
    height = int(target_height * height_ratio)
    
    # Asegurarse de que las coordenadas estén dentro de los límites de la imagen
    start_x = max(0, min(start_x, img_width - 1))
    start_y = max(0, min(start_y, img_height - 1))
    width = min(width, img_width - start_x)
    height = min(height, img_height - start_y)
    
    # Si el área de recorte es demasiado pequeña, usar la imagen completa
    if width < 50 or height < 50:
        print(f"Imagen demasiado pequeña para recortar: {img_width}x{img_height}")
        return cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    
    # Realizar el recorte
    end_x = start_x + width
    end_y = start_y + height
    cropped_img = img[start_y:end_y, start_x:end_x]
    
    # Redimensionar la imagen recortada
    return cv2.resize(cropped_img, (IMG_SIZE[1], IMG_SIZE[0]))


# Update the visualization function to show the exact crop
def visualize_preprocessing(image_path, save_path=None):
    """
    Visualize how the preprocessing affects the input images using exact crop coordinates.
    
    Args:
        image_path: Path to a sample image
        save_path: Optional path to save the visualization
    """
    # Read the original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Could not read image at {image_path}")
        return None, None, None, None
        
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Use exact coordinates for cropping
    start_x = 943
    start_y = 861
    width = 3569
    height = 1651
    
    # Ensure coordinates are within image boundaries
    img_height, img_width = original_img.shape[:2]
    
    # Print original image dimensions for reference
    print(f"Original image dimensions: {img_width}x{img_height}")
    
    # Adjust if necessary to prevent out-of-bounds errors
    start_x = min(start_x, img_width - 1)
    start_y = min(start_y, img_height - 1)
    width = min(width, img_width - start_x)
    height = min(height, img_height - start_y)
    
    end_x = start_x + width
    end_y = start_y + height
    
    # Create a copy of the original image with the ROI highlighted
    highlighted_img = original_img.copy()
    cv2.rectangle(highlighted_img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 5)
    
    # Crop the image
    cropped_img = original_img[start_y:end_y, start_x:end_x]
    
    # Resize the cropped image to model input size
    resized_img = cv2.resize(cropped_img, (IMG_SIZE[1], IMG_SIZE[0]))
    
    # Create the figure for visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot each step
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(highlighted_img)
    axes[1].set_title(f'Region of Interest\n({start_x},{start_y}, {width}x{height})')
    axes[1].axis('off')
    
    axes[2].imshow(cropped_img)
    axes[2].set_title('Cropped Image')
    axes[2].axis('off')
    
    axes[3].imshow(resized_img)
    axes[3].set_title(f'Resized to {IMG_SIZE[0]}x{IMG_SIZE[1]}')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return original_img, highlighted_img, cropped_img, resized_img

# Test the preprocessing function with a sample image

def test_with_sample_image():
    """
    Test the preprocessing with a sample image to verify the crop coordinates.
    """
    # Find a sample image
    sample_paths = []
    for root, _, files in os.walk(TRAIN_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_paths.append(os.path.join(root, file))
                break
        if sample_paths:
            break
    
    if not sample_paths:
        print("No sample images found for testing.")
        return
    
    sample_path = sample_paths[0]
    print(f"Testing with sample image: {sample_path}")
    
    # Visualize the preprocessing
    visualize_preprocessing(sample_path, 'crop_test.png')
    
    print("Test complete.")

# Run the test
test_with_sample_image()




# Calculate parameters based on directory size
train_samples = sum([len(files) for _, _, files in os.walk(TRAIN_DIR)])
validation_samples = sum([len(files) for _, _, files in os.walk(VALIDATION_DIR)])
BATCH_SIZE = 16  # Reduced batch size for higher resolution
steps_per_epoch = train_samples // BATCH_SIZE
validation_steps = validation_samples // BATCH_SIZE

# Data generator for training with data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,  # Custom preprocessing applied first
    rescale=1./255,    #Normalize here
    rotation_range=5,           # Reduced rotation for sludge images
    width_shift_range=0.1,       # Horizontal shift
    height_shift_range=0.1,      # Vertical shift
    # shear_range=0.1,             # Reduced shear
    zoom_range=0.15,             # Zoom range
    horizontal_flip=True,        # Horizontal flip
    fill_mode='nearest'          # Fill mode for augmentation
)

# # Data generator for training with data augmentation
# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_image,  # Custom preprocessing applied first
#     rescale=1./255,    #Normalize here
#     rotation_range=15,           # Reduced rotation for sludge images
#     width_shift_range=0.1,       # Horizontal shift
#     height_shift_range=0.1,      # Vertical shift
#     shear_range=0.1,             # Reduced shear
#     zoom_range=0.15,             # Zoom range
#     horizontal_flip=True,        # Horizontal flip
#     fill_mode='nearest'          # Fill mode for augmentation
# )

# Data generator for validation (only preprocessing, no augmentation)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image, # Only apply the custom preprocessing
    rescale=1./255,    #Normalize here

)

# Prepare the data generators with the specified target size
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,  # This will be applied after preprocessing
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,  # This will be applied after preprocessing
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Get class indices from the generator
class_indices = train_generator.class_indices
print(f"Class indices: {class_indices}")

# Verify preprocessing and normalization
print("Verifying preprocessing:")
batch_x, batch_y = next(train_generator)
print(f"Batch shape: {batch_x.shape}")  # Should be (BATCH_SIZE, HEIGHT, WIDTH, 3)
print(f"Value range: [{batch_x.min()}, {batch_x.max()}]")  # Should be close to [0, 1]

# Display a few processed training images to verify preprocessing
plt.figure(figsize=(15, 5))
for i in range(min(5, batch_x.shape[0])):
    plt.subplot(1, 5, i+1)
    plt.imshow(batch_x[i])
    plt.title(f"Class: {int(batch_y[i])}")
    plt.axis('off')
plt.suptitle("Processed Training Images")
plt.tight_layout()
plt.show()

# Calculate class weights to handle imbalance
class_counts = [0, 0]
for i in range(len(train_generator.classes)):
    class_counts[train_generator.classes[i]] += 1

print(f"Class distribution in training set: {class_counts}")

# Calculate weights inversely proportional to class frequency
if class_counts[0] != class_counts[1]:
    total = sum(class_counts)
    class_weight = {
        0: total / (2 * class_counts[0]),
        1: total / (2 * class_counts[1])
    }
    print(f"Calculated class weights: {class_weight}")
else:
    class_weight = None
    print("Classes are balanced. No class weights needed.")

# # # 3.3 Definir y construir el modelo base
# # # ---------------------------------------------

def build_model(base_model_name='EfficientNetB0', input_shape=(224, 224, 3), 
                trainable_base=False, dropout_rate=0.3):
    """
    Construye un modelo para detección de anomalías basado en redes pre-entrenadas.
    
    Args:
        base_model_name: Nombre del modelo base ('MobileNetV2', 'ResNet50', o 'EfficientNetB0')
        input_shape: Forma de la entrada (altura, ancho, canales)
        trainable_base: Si es True, hace que las capas base sean entrenables
        dropout_rate: Tasa de dropout para regularización
    
    Returns:
        Modelo compilado
    """
    # Crear el modelo base
    if base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                               input_shape=input_shape)
        #######################################
    # Nuevo modeloa probar
    elif base_model_name == 'MobileNetV3Small':
        base_model =tf.keras.applications.MobileNetV3Small(input_shape=input_shape,
        include_top=False,
        weights='imagenet')
         # Nuevo modeloa probar
    elif base_model_name == 'MobileNetV3Large':
        base_model =tf.keras.applications.MobileNetV3Large(input_shape=input_shape,
        include_top=False,
        weights='imagenet')
    elif base_model_name == 'EfficientNetB2':
        # Importar y cargar EfficientNetB2 con pesos de ImageNet
        base_model = tf.keras.applications.EfficientNetB2(weights='imagenet', 
                                                         include_top=False,  
                                                         input_shape=input_shape)
##############################################
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, 
                            input_shape=input_shape)
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                                  input_shape=input_shape)
    else:
        raise ValueError(f"Modelo base no soportado: {base_model_name}")
    
    # Congelar o descongelar las capas base
    base_model.trainable = trainable_base
    
    # Añadir capas superiores personalizadas
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compilar el modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    
    return model



# Analizador detallado de las imágenes procesadas
def analyze_batch_images(batch_x, batch_y, num_samples=5):
    """
    Analiza detalladamente las imágenes procesadas para diagnosticar
    problemas de visualización.
    """
    print("\n--- ANÁLISIS DETALLADO DE IMÁGENES ---")
    
    # Estadísticas globales del batch
    print(f"Shape del batch: {batch_x.shape}")
    print(f"Dtype del batch: {batch_x.dtype}")
    print(f"Rango global: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
    
    # Crear figura para visualización
    plt.figure(figsize=(18, 12))
    
    for i in range(min(num_samples, batch_x.shape[0])):
        img = batch_x[i]
        label = batch_y[i]
        
        # Estadísticas de la imagen
        print(f"\nImagen {i+1} (Clase {int(label)}):")
        print(f"  Rango: [{img.min():.4f}, {img.max():.4f}]")
        print(f"  Media: {img.mean():.4f}")
        print(f"  Desviación estándar: {img.std():.4f}")
        
        # Histograma de valores
        values_flat = img.reshape(-1)
        
        # Visualización original
        plt.subplot(num_samples, 4, i*4 + 1)
        plt.imshow(img)
        plt.title(f"Original (Clase {int(label)})")
        plt.axis('off')
        
        # Histograma
        plt.subplot(num_samples, 4, i*4 + 2)
        plt.hist(values_flat, bins=50, color='blue', alpha=0.7)
        plt.title(f"Histograma\nRango: [{img.min():.2f}, {img.max():.2f}]")
        plt.grid(True, alpha=0.3)
        
        # Visualización con contraste ajustado
        plt.subplot(num_samples, 4, i*4 + 3)
        plt.imshow(img, vmin=0.0, vmax=0.5)  # Ajustar máximo para mejor visualización
        plt.title("Contraste ajustado\nvmax=0.5")
        plt.axis('off')
        
        # Versión CLAHE para comparación
        img_uint8 = (img * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        plt.subplot(num_samples, 4, i*4 + 4)
        plt.imshow(enhanced_img)
        plt.title("CLAHE aplicado")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# Obtener un nuevo batch para analizar
batch_x, batch_y = next(train_generator)

# Ejecutar el análisis detallado
analyze_batch_images(batch_x, batch_y)



#########################################################################################################################################




#AQUI LO LLAMAAS
# Construir el modelo con MobileNetV2 como base
print("Construyendo modelo base con MobileNetV3Small...")
model = build_model(base_model_name='MobileNetV3Small', 
                   input_shape=IMG_SIZE + (3,),
                   trainable_base=False)

# Mostrar el resumen del modelo
model.summary()

# # 3.4 Definir callbacks para el entrenamiento
# # ---------------------------------------------

# Directorio para logs de TensorBoard

#Por termnial le pasas esto: 
# 
# tensorboard --logdir=../logs

# y luego en el navegador;
# http://localhost:6006/

log_dir = os.path.join("../logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)

# Definir callbacks
callbacks = [
    # Guardar el mejor modelo
    ModelCheckpoint(
        filepath=os.path.join(FINE_TUNED_DIR, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Detener el entrenamiento si no hay mejora
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Reducir la tasa de aprendizaje cuando el rendimiento se estanca
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    # TensorBoard para monitoreo visual
    TensorBoard(log_dir=log_dir, histogram_freq=1)  
]

# 3.5 Entrenar el modelo (primera fase - solo capas superiores)
# ---------------------------------------------

# Número de épocas para entrenamiento
EPOCHS = 40

print("Iniciando entrenamiento de capas superiores...")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    class_weight=class_weight,  # NUEVO: Usar class_weight
    verbose=1
)

# 3.6 Fine-tuning (segunda fase - incluir algunas capas del modelo base)
# ---------------------------------------------

print("Iniciando fine-tuning con algunas capas del modelo base...")

# Guardar los pesos entrenados hasta ahora
model.save(os.path.join(FINE_TUNED_DIR, 'model_phase1.h5'))

# Ahora vamos a descongelar algunas capas del modelo base
if isinstance(model.layers[1], tf.keras.Model):  # Si la capa base es un modelo
    base_model = model.layers[1]
    
    # Congelar las primeras capas y descongelar las últimas
    # Para EfficientNetB0, que tiene múltiples bloques
    # Descongelamos solo los últimos bloques
    for layer in base_model.layers:
        layer.trainable = False
    
    # Descongelar los últimos 5 layers
    for layer in base_model.layers[-5:]:
        layer.trainable = True
    
    # Usar una tasa de aprendizaje más baja para fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    
    # Mostrar el modelo actualizado
    model.summary()
    
    # Entrenar con fine-tuning
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,  # Menos épocas para fine-tuning
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight,  # NUEVO: Usar class_weight
        verbose=1
    )
    
    # Combinar historiales de entrenamiento
    total_history = {}
    for k in history.history.keys():
        if k in history_fine.history:
            total_history[k] = history.history[k] + history_fine.history[k]
else:
    print("No se pudo realizar fine-tuning en el modelo base")
    total_history = history.history

# 3.7 Visualizar resultados del entrenamiento
# ---------------------------------------------

def plot_training_history(history_obj, metrics=['accuracy', 'loss']):
    """
    Visualiza las métricas de entrenamiento.
    
    Args:
        history_obj: Historial de entrenamiento o diccionario con las métricas
        metrics: Lista de métricas para visualizar
    """
    plt.figure(figsize=(15, 10))
    
    # Verificar el tipo de objeto historial
    if hasattr(history_obj, 'history'):
        # Si es un objeto History de Keras
        history_dict = history_obj.history
    else:
        # Si ya es un diccionario
        history_dict = history_obj
    
    print("Métricas disponibles:", list(history_dict.keys()))
    
    for i, metric in enumerate(metrics):
        if metric in history_dict:
            plt.subplot(2, 2, i+1)
            plt.plot(history_dict[metric], label=f'Training {metric}')
            
            # Verificar si existe la métrica de validación
            val_metric = f'val_{metric}'
            if val_metric in history_dict:
                plt.plot(history_dict[val_metric], label=f'Validation {metric}')
            
            plt.title(f'{metric.capitalize()} Over Time')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
        else:
            print(f"Advertencia: Métrica '{metric}' no encontrada en el historial")
    
    plt.tight_layout()
    plt.show()

# Visualizar el historial de entrenamiento
metrics_to_plot = ['accuracy', 'loss', 'precision', 'recall']

try:
    print("Visualizando historial combinado...")
    plot_training_history(total_history, metrics=metrics_to_plot)
except NameError:
    try:
        print("Visualizando historial de la primera fase...")
        plot_training_history(history, metrics=metrics_to_plot)
    except NameError:
        print("No hay historial de entrenamiento disponible para visualizar.")

# # # 3.8 Guardar el modelo final
# # # ---------------------------------------------

# Guardar el modelo completo
model.save(os.path.join(FINE_TUNED_DIR, 'final_model.h5'))

# Guardar también en formato TensorFlow SavedModel para implementación
model.save(os.path.join(FINE_TUNED_DIR, 'saved_model.keras'))

# Guardar los mapeos de clase
class_indices = train_generator.class_indices
with open(os.path.join(FINE_TUNED_DIR, 'class_indices.json'), 'w') as f:
    json.dump(class_indices, f)

# NUEVO: Guardar también el mejor umbral para clasificación
# Evaluamos en el conjunto de validación y buscamos el umbral óptimo
print("\nBuscando umbral óptimo para clasificación...")

validation_generator.reset()
y_true = []
y_pred = []

for i in range(validation_steps):
    x, y = next(validation_generator)
    pred = model.predict(x, verbose=0)
    y_true.extend(y)
    y_pred.extend(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Encontrar el mejor umbral usando F1 score
from sklearn.metrics import f1_score

best_threshold = 0.5
best_f1 = 0.0

thresholds = np.arange(0.1, 0.9, 0.05)
for threshold in thresholds:
    y_pred_binary = (y_pred >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred_binary)
    print(f"Umbral: {threshold:.2f}, F1 score: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor umbral encontrado: {best_threshold:.4f} con F1 score: {best_f1:.4f}")

# Guardar el mejor umbral
with open(os.path.join(FINE_TUNED_DIR, 'best_threshold.txt'), 'w') as f:
    f.write(str(best_threshold))

print(f"Modelo guardado en {os.path.join(FINE_TUNED_DIR, 'final_model.h5')}")
print(f"Modelo SavedModel guardado en {os.path.join(FINE_TUNED_DIR, 'saved_model')}")
print(f"Índices de clase guardados en {os.path.join(FINE_TUNED_DIR, 'class_indices.json')}")
print(f"Mejor umbral guardado en {os.path.join(FINE_TUNED_DIR, 'best_threshold.txt')}")

print("Fine-tuning completado con éxito!")