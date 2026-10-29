import tensorflow as tf
import cv2
import numpy as np

cnn_model = tf.keras.models.load_model("models/accident_cnn.h5")

def is_accident(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = cnn_model.predict(img, verbose=0)[0][0]

    # 🔥 LOWER threshold
    return pred > 0.4

