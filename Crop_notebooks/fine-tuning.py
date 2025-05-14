import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import datetime
import json
from sklearn.metrics import f1_score

# Import preprocessing functions
from Preprocessing import (
    preprocess_image, 
    IMG_SIZE, 
    get_data_generators
)

# Define paths
MODELS_DIR = 'models/'
FINE_TUNED_DIR = os.path.join(MODELS_DIR, 'fine_tuned')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FINE_TUNED_DIR, exist_ok=True)

def build_model(base_model_name='EfficientNetB0', input_shape=(186, 400, 3), 
                trainable_base=False, dropout_rate=0.3):
    """
    Builds a model for anomaly detection based on pre-trained networks.
    
    Args:
        base_model_name: Name of the base model ('MobileNetV2', 'ResNet50', 'EfficientNetB0', etc.)
        input_shape: Input shape (height, width, channels)
        trainable_base: If True, makes base layers trainable
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled model
    """
    # Create the base model
    if base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                               input_shape=input_shape)
    # New models to try
    elif base_model_name == 'MobileNetV3Small':
        base_model =tf.keras.applications.MobileNetV3Small(input_shape=input_shape,
        include_top=False,
        weights='imagenet')
    elif base_model_name == 'MobileNetV3Large':
        base_model =tf.keras.applications.MobileNetV3Large(input_shape=input_shape,
        include_top=False,
        weights='imagenet')
    elif base_model_name == 'EfficientNetB2':
        # Import and load EfficientNetB2 with ImageNet weights
        base_model = tf.keras.applications.EfficientNetB2(weights='imagenet', 
                                                         include_top=False,  
                                                         input_shape=input_shape)
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, 
                            input_shape=input_shape)
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                                  input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # Freeze or unfreeze base layers
    base_model.trainable = trainable_base
    
    # Add custom top layers
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
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    
    return model

def plot_training_history(history_obj, metrics=['accuracy', 'loss']):
    """
    Visualizes training metrics.
    
    Args:
        history_obj: Training history or dictionary with metrics
        metrics: List of metrics to visualize
    """
    plt.figure(figsize=(15, 10))
    
    # Verify the type of history object
    if hasattr(history_obj, 'history'):
        # If it's a Keras History object
        history_dict = history_obj.history
    else:
        # If it's already a dictionary
        history_dict = history_obj
    
    print("Available metrics:", list(history_dict.keys()))
    
    for i, metric in enumerate(metrics):
        if metric in history_dict:
            plt.subplot(2, 2, i+1)
            plt.plot(history_dict[metric], label=f'Training {metric}')
            
            # Check if validation metric

def find_optimal_threshold(model, validation_generator, validation_steps):
    """
    Encuentra el umbral óptimo para la clasificación binaria.
    
    Args:
        model: Modelo entrenado
        validation_generator: Generador de datos de validación
        validation_steps: Número de pasos para validación
        
    Returns:
        El mejor umbral encontrado y su F1 score
    """
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
    return best_threshold, best_f1

def save_model_and_metadata(model, class_indices, best_threshold, model_dir):
    """
    Guarda el modelo y los metadatos relacionados.
    
    Args:
        model: Modelo entrenado
        class_indices: Índices de clase del generador
        best_threshold: Mejor umbral para clasificación
        model_dir: Directorio donde guardar el modelo
    """
    # Guardar el modelo completo
    model.save(os.path.join(model_dir, 'final_model.h5'))

    # Guardar también en formato TensorFlow SavedModel para implementación
    model.save(os.path.join(model_dir, 'saved_model.keras'))

    # Guardar los mapeos de clase
    with open(os.path.join(model_dir, 'class_indices.json'), 'w') as f:
        json.dump(class_indices, f)

    # Guardar el mejor umbral
    with open(os.path.join(model_dir, 'best_threshold.txt'), 'w') as f:
        f.write(str(best_threshold))

    print(f"Modelo guardado en {os.path.join(model_dir, 'final_model.h5')}")
    print(f"Modelo SavedModel guardado en {os.path.join(model_dir, 'saved_model')}")
    print(f"Índices de clase guardados en {os.path.join(model_dir, 'class_indices.json')}")
    print(f"Mejor umbral guardado en {os.path.join(model_dir, 'best_threshold.txt')}")

def train_model(model_name='MobileNetV3Small', epochs=40, batch_size=16):
    """
    Entrena el modelo completo, incluido fine-tuning.
    
    Args:
        model_name: Nombre del modelo base a usar
        epochs: Número de épocas para entrenar
        batch_size: Tamaño del lote
        
    Returns:
        Historial de entrenamiento combinado
    """
    # Obtener generadores de datos
    (train_generator, validation_generator, 
     class_indices, class_weight, 
     steps_per_epoch, validation_steps) = get_data_generators(
        preprocessing_func=preprocess_image, 
        batch_size=batch_size
    )
    
    # Directorio para logs de TensorBoard
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
    
    # Construir el modelo
    print(f"Construyendo modelo base con {model_name}...")
    model = build_model(
        base_model_name=model_name, 
        input_shape=IMG_SIZE + (3,),
        trainable_base=False
    )
    
    # Mostrar el resumen del modelo
    model.summary()
    
    # Primera fase: entrenar solo las capas superiores
    print("Iniciando entrenamiento de capas superiores...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # Guardar los pesos entrenados hasta ahora
    model.save(os.path.join(FINE_TUNED_DIR, 'model_phase1.h5'))
    
    # Segunda fase: fine-tuning con algunas capas del modelo base
    print("Iniciando fine-tuning con algunas capas del modelo base...")
    
    # Now let's unfreeze some layers of the base model
    if isinstance(model.layers[1], tf.keras.Model):  # Si la capa base es un modelo
        base_model = model.layers[1]
        
        # Congelar las primeras capas y descongelar las últimas
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
            class_weight=class_weight,
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
    
    # Encontrar el umbral óptimo para clasificación
    best_threshold, _ = find_optimal_threshold(model, validation_generator, validation_steps)
    
    # Guardar el modelo y metadatos
    save_model_and_metadata(model, class_indices, best_threshold, FINE_TUNED_DIR)
    
    return total_history

if __name__ == "__main__":
    # Configuración inicial
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Entrenar el modelo
    history = train_model(model_name='MobileNetV2', epochs=40, batch_size=16)
    
    # Visualizar el historial de entrenamiento
    metrics_to_plot = ['accuracy', 'loss', 'precision', 'recall']
    plot_training_history(history, metrics=metrics_to_plot)
    
    print("Fine-tuning completado con éxito!")