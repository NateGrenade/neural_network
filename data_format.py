import tensorflow as tf
import numpy as np


inputs = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(inputs)
x1 = tf.keras.layers.Dense(128, activation='relu')(x)
x2 = tf.keras.layers.Dense(128, activation='relu')(x1)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x2)

activation_model = tf.keras.Model(inputs=inputs, outputs=[x1, x2, outputs])

# Load MNIST test image to test activations
(_, _), (img_test, _) = tf.keras.datasets.mnist.load_data()
img_test = tf.keras.utils.normalize(img_test, axis=1)

input_image = img_test[0].reshape(1, 28, 28)  # Single input

# Get activations
activations = activation_model.predict(input_image)

# Unpack activations
l1_activations = activations[0]  # Stores activations of first hidden layer
l2_activations = activations[1]  # Stores activations of second hidden layer
out_activations = activations[2]  # Stores activations of output layer

prediction = np.argmax(out_activations)
confidence = out_activations[0][prediction]

print(prediction)
print(f"Confidence: {confidence:.4f}")
