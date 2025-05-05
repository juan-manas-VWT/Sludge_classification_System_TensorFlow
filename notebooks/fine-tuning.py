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

# 3.2 Preparar generadores de datos
# ---------------------------------------------
# Definir parámetros directamente en el script
IMG_SIZE = (448, 640)
BATCH_SIZE = 16 # Reducir de 32 al usar resoluciones mayores

# Calcular parámetros basados en el directorio
train_samples = sum([len(files) for _, _, files in os.walk(TRAIN_DIR)])
validation_samples = sum([len(files) for _, _, files in os.walk(VALIDATION_DIR)])
steps_per_epoch = train_samples // BATCH_SIZE
validation_steps = validation_samples // BATCH_SIZE
# MODIFICACIÓN IMPORTANTE: Añadir rescale=1./255 para normalizar aquí
# Generador para entrenamiento con aumento de datos y normalización
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Normalización aquí
    rotation_range=20,           
    width_shift_range=0.1,       # Aumento de datos
    height_shift_range=0.1,      # Aumento de datos
    shear_range=0.2,             # Aumento de datos
    zoom_range=0.2,              # Aumento de datos
    horizontal_flip=True,        # Aumento de datos
    ##########
    # brightness_range=[0.8, 1.2],  # Variaciones de brillo
    # channel_shift_range=0.1,      # Cambios en canales de color
    ##########
    fill_mode='nearest'
)

# Generador para validación con normalización
val_datagen = ImageDataGenerator(
    rescale=1./255              # Normalización aquí
)

# Preparar los generadores de flujo de datos
# MODIFICACIÓN IMPORTANTE: Añadir target_size=IMG_SIZE para hacer el resize aquí
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,       # Resize aquí
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,       # Resize aquí
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)
# Obtener índices de clase del generador
class_indices = train_generator.class_indices
print(f"Índices de clase: {class_indices}")

# NUEVO: Verificar el rango de valores de las imágenes
batch_x, batch_y = next(train_generator)
print(f"Verificación de normalización:")
print(f"Rango de valores en batch: [{batch_x.min()}, {batch_x.max()}]")  # Debe estar cerca de [0, 1]
print(f"Forma del batch: {batch_x.shape}")  # Debe ser (BATCH_SIZE, HEIGHT, WIDTH, 3)  (32, 224, 224, 3)

# NUEVO: Calcular class weights para manejar desbalance
class_counts = [0, 0]
for i in range(len(train_generator.classes)):
    class_counts[train_generator.classes[i]] += 1

print(f"Distribución de clases en entrenamiento: {class_counts}")  #para saber el total de clases

# Calcular pesos inversamente proporcionales a la frecuencia
if class_counts[0] != class_counts[1]:
    total = sum(class_counts)
    class_weight = {
        0: total / (2 * class_counts[0]),
        1: total / (2 * class_counts[1])
    }
    print(f"Class weights calculados: {class_weight}")
else:
    class_weight = None
    print("Las clases están balanceadas. No se necesitan class weights.")

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

#AQUI LO LLAMAAS
# Construir el modelo con MobileNetV2 como base
print("Construyendo modelo base con EfficientNetB2...")
model = build_model(base_model_name='EfficientNetB2', 
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