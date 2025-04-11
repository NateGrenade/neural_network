import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data_set = tf.keras.datasets.mnist
# there exists a premade, and rather large pool of handwritten digits with labels to pull from

(img_train, lbl_train),(img_test,lbl_test) = data_set.load_data()
# pulls the data from the data set into four variables, two holding image pixel data and two holding labels for the data

img_train = tf.keras.utils.normalize(img_train, axis=1)
img_test = tf.keras.utils.normalize(img_test, axis=1)
# itemizes and "flattens" the 2D array of pixel data into a one dimensional array (vector)

model = tf.keras.models.Sequential()
# creates the specific model that will be trained on the data

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
#

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(img_train, lbl_train, epochs=3)

model.save('model/digit_recognizer.keras')

