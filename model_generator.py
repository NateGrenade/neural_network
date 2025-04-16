import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

MODEL_PATH = "model/digit_recognizer.keras"


def train(layer_total, nodes):
    try:
        inputs = tf.keras.Input(shape=(28, 28), name="input_layer")
        #
        x = tf.keras.layers.Flatten(name="flatten")(inputs)
        for i in range(layer_total - 1):
            x = tf.keras.layers.Dense(nodes[i], activation='relu', name=f"dense_{i + 1}")(x)

            #defines the hidden layers of the network
        outputs = tf.keras.layers.Dense(10, activation='softmax', name="output")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Loads the data, then normalizes it to between 0-1
        data_set = tf.keras.datasets.mnist
        (img_train, lbl_train), (img_test, lbl_test) = data_set.load_data()
        img_train = tf.keras.utils.normalize(img_train, axis=1)
        img_test = tf.keras.utils.normalize(img_test, axis=1)

        # Compiles the model, then trains it on the premade dataset of handwritten images
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Starting training...")
        model.fit(img_train, lbl_train, epochs=5, verbose=1)#loops through the training data 5 times

        # Saves model in a new file under the model folder
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error occurred while training or saving: {e}")
        raise


if __name__ == "__main__":
    try:
        train(3, [128, 128])
    except Exception as e:
        print(f"Main execution failed: {e}")