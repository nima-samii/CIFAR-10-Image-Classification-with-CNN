from tensorflow.keras.datasets import cifar10 # type: ignore
import keras
import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation # type: ignore


def load_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("\n---------------------------------------------------")
    print("Load dataset...\n")
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print("---------------------------------------------------")

    return x_train, y_train, x_test, y_test


def preprocess_dataset(x_train, y_train, x_test, y_test, num_classes=10):
    print("Normalization and one hot encoding ...")
    # Normalize the data. Before we need to connvert data type to float for computation.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Convert class vectors to binary class matrices. This is called one hot encoding.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print("---------------------------------------------------")

    return x_train, y_train, x_test, y_test


def data_augmentation(x_train):
    """Applies data augmentation to the training dataset."""
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),  # Randomly flip images horizontally
        RandomRotation(0.1),       # Rotate images by 10% of 180 degrees
        RandomZoom(0.1),           # Zoom in/out by 10%
        RandomTranslation(0.1, 0.1) # Shift images up/down and left/right by 10%
    ])
    
    return data_augmentation(x_train, training=True)  # Apply augmentation