import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Configuración de hiperparámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_WARMUP = 5
EPOCHS_FINETUNE = 15
DATA_DIR = "dataset/"  # Ruta a la carpeta con las subcarpetas de cada clase

# 1. Carga del Dataset (Dividido en entrenamiento y validación)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Guardar los nombres de las clases en un JSON
class_names = train_ds.class_names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Optimización del rendimiento de carga
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. Pipeline de Data Augmentation para corregir sesgos visuales (reflejos y brillos)
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.3),  # Modifica brillos para no confundir vidrio/plástico
        layers.RandomBrightness(0.2),
    ]
)

# 3. Construcción del Modelo Base MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # Congelado inicialmente

# Arquetipo del modelo final
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# 4. Fase 1: Entrenamiento inicial de las capas finales
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("--- Fase 1: Entrenamiento de capas clasificadoras ---")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_WARMUP)

# 5. Fase 2: Fine-Tuning (Descongelar las últimas 40 capas de MobileNetV2)
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Recompilar con un Learning Rate bajo para evitar destruir los pesos preentrenados
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("--- Fase 2: Fine-Tuning de capas profundas ---")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE)

# 6. Guardar el modelo corregido
model.save("waste_mobilenet.keras")
print("✅ Modelo corregido guardado como 'waste_mobilenet.keras'")
