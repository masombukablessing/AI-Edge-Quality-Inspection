import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

# Paths
DATASET_PATH = r"C:\Users\csgadm#\OneDrive\Desktop\dataset"
SAVED_MODEL_PATH = 'saved_model/quality_model.h5'

# Image data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128,128),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128,128),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# Build model
model = build_model()

# Train model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Save trained model
os.makedirs('saved_model', exist_ok=True)
model.save(SAVED_MODEL_PATH)
print(f"Model saved at {SAVED_MODEL_PATH}")

