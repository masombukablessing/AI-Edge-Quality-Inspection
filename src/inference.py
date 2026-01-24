# Image inference code
import cv2
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("models/model.h5")

# Read test image
image_path = "results/sample.jpg"  # change image name if needed
img = cv2.imread(image_path)

# Resize and normalize
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)[0][0]

# Output result
if prediction > 0.5:
    print("✅ GOOD PRODUCT")
else:
    print("❌ DEFECTIVE PRODUCT")
