import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from model_generator import img_test, lbl_test

model = tf.keras.models.load_model('model/digit_recognizer.keras')

loss, accuracy = model.evaluate(img_test,lbl_test)

print("Accuracy: "+f"{accuracy}")