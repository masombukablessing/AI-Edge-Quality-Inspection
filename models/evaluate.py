import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from model import build_model

SAVED_MODEL_PATH = 'saved_model/quality_model.h5'
DATASET_PATH = r"C:\Users\csgadm#\OneDrive\Desktop\dataset"

# Load model
model = tf.keras.models.load_model(SAVED_MODEL_PATH)

# Example: predict a single image
img_path = os.path.join(DATASET_PATH, 'defective_images', 'example1.jpg')
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)/255.0
img_array = tf.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_idx = tf.argmax(prediction[0]).numpy()

classes = ['defective', 'good']
print(f"Predicted class: {classes[class_idx]}")

