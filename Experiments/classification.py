import tensorflow as tf
import os
from Models.CNN import CNN_Model
from Utils import utils
import numpy as np
import random
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore


class Experiment_Classification():
    def __init__(self,
                 epochs=10,
                 batch_size=32,
                 num_classes=10,
                 network_structure='CNN',
                 data_augmentation = False,
                 base_project_dir='.'):
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.data_augmentation = data_augmentation
        self.network_structure = network_structure
        self.base_project_dir = base_project_dir
        self.task = 'classification_' + self.network_structure

        self.output_dir = os.path.join(self.base_project_dir, 'Output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.output_models_dir = os.path.join(
            self.base_project_dir, 'Output_Models')
        if not os.path.exists(self.output_models_dir):
            os.mkdir(self.output_models_dir)

        self.output_models_dir = os.path.join(
            self.output_models_dir, self.task)
        if not os.path.exists(self.output_models_dir):
            os.mkdir(self.output_models_dir)

        print(self.output_dir, self.output_models_dir)
        print("\n---------------------------------------------------")

        # Set the random seed for reproducibility
        random.seed(42)  # Python's random module seed
        np.random.seed(42)  # NumPy's random module seed
        tf.random.set_seed(42)  # TensorFlow's random seed

        # self.fold_history = FoldsHistory()
        self.prepare()
        if self.network_structure == 'CNN':
            self.model = CNN_Model(input_shape=self.input_shape, num_classes=self.num_classes)

        self.model.summary(expand_nested=True)
        self.time = datetime.datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")

        if not os.path.exists("logs"):
            os.makedirs("logs")

        checkpoint_path = os.path.join(self.output_models_dir, "best_model.h5")
        self.checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,  
            monitor='val_accuracy',  # Track validation accuracy
            save_best_only=True,  # Only save the best model
            save_weights_only=False,  # Save entire model
            verbose=1
        )

    def __create_model(self, learning_rate=0.0001, decay=1e-6):
        # Initiate RMSprop optimizer
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, decay=decay)

        model = tf.keras.models.clone_model(self.model)
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def prepare(self):

        self.x_train, self.y_train, self.x_test, self.y_test = utils.preprocess_dataset(*utils.load_dataset())

        self.input_shape = self.x_train.shape[1:]
        self.num_classes = self.y_train.shape[1]


    def train(self):

        self.history = None  # For recording the history of trainning process.
        if not self.data_augmentation:
            print("\n---------------------------------------------------")
            print('Not using data augmentation...')
            model = self.__create_model()
            self.history = model.fit(self.x_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.x_test, self.y_test),
            shuffle=True,
            callbacks=[self.checkpoint] )
        
        else:
            print("\n---------------------------------------------------")
            print('Using data augmentation...')
            # Apply augmentation to the training dataset
            x_train_augmented = utils.data_augmentation(self.x_train)

            model = self.__create_model()
            self.history = model.fit(
                x_train_augmented, self.y_train,  # Use augmented data
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(self.x_test, self.y_test),
                shuffle=True,
                callbacks=[self.checkpoint] 
            )            

            # print('Using real-time data augmentation...')
    
            # train_dataset = (
            #     tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
            #     .shuffle(len(self.x_train))
            #     .batch(self.batch_size)
            #     .map(lambda x, y: (utils.data_augmentation(x, training=True), y))
            #     .prefetch(tf.data.AUTOTUNE)
            # )
            
            # test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(self.batch_size)

            # model = self.__create_model()
            
            # self.history = model.fit(
            #     train_dataset,
            #     epochs=self.epochs,
            #     validation_data=test_dataset
            # )