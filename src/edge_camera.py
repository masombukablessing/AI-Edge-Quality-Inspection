# Real-time edge camera inference
import cv2
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("models/model.h5")

# Open camera (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("📷 Camera started. Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        label = "GOOD PRODUCT"
        color = (0, 255, 0)
    else:
        label = "DEFECTIVE PRODUCT"
        color = (0, 0, 255)

    # Display result
    cv2.putText(
        frame,
        label,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Edge AI Quality Inspection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Camera stopped")
