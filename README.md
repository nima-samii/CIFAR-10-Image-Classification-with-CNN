# CIFAR-10 Image Classification

This project implements a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset using TensorFlow and Keras. It includes dataset preprocessing, data augmentation, model training, and model checkpointing to save the best performing model.

## Project Structure

```
.
├── Utils
│   ├── utils.py              # Dataset loading, preprocessing, and data augmentation
│
├── Models
│   ├── CNN.py                # CNN model definition
│
├── Experiments
│   ├── classification.py     # Training and evaluation script
│
├── Output                    # Stores training logs and results
│
├── Output_Models             # Stores saved models
│
├── logs                      # Stores TensorBoard logs
│
└── README.md                 # Project documentation
```

## Features

✅ **Dataset Handling**: Loads CIFAR-10 dataset, normalizes images, and applies one-hot encoding to labels.\
✅ **CNN Model**: A multi-layer convolutional neural network is implemented in `Models/CNN.py`.\
✅ **Data Augmentation**: Optional augmentation includes flipping, rotation, zooming, and translation.\
✅ **Model Training**: Trains the model with or without data augmentation and tracks validation accuracy.\
✅ **Model Checkpointing**: Saves the best model during training to prevent performance degradation.


## Model Training

- The script allows training with or without data augmentation.
- It saves the best model based on validation accuracy to `Output_Models/classification_CNN/best_model.h5`.
- The best model is stored and can be used for further inference or fine-tuning.

### Final Model Performance

- **Training Accuracy**: 66.52%
- **Validation Accuracy**: 69.15%
- **Training Loss**: 0.9533
- **Validation Loss**: 0.8910
