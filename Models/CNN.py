
import tensorflow as tf


def CNN_Model(input_shape=(32, 32, 3), num_classes=10):
    # Define the convnet
    model = tf.keras.Sequential([
        # Block 1: CONV => RELU => CONV => RELU => POOL => DROPOUT
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3, 3)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block 2: CONV => RELU => CONV => RELU => POOL => DROPOUT
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block 3: FLATTEN => DENSE => RELU => DROPOUT
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),

        # Output layer with softmax activation
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.Activation('softmax')
    ])

    return model