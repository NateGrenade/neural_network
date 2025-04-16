import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model_generator import MODEL_PATH

(_,_),(test_images, _) = tf.keras.datasets.mnist.load_data()
test_images = test_images/255.0

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    prediction = model.predict(test_images[0:1])
    predicted_digit = np.argmax(prediction[0])
    print(f"Predicted digit: {predicted_digit}")
except Exception as e:
    print(f"Yikers... {e}")