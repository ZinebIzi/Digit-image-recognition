# Digit-image-recognition

## Overview
This project aims to implement and explore different neural network architectures using the MNIST dataset for handwritten digit recognition. Starting from a simple perceptron, we progress to a shallow network, a deep network, and finally implement LeNet-5, a convolutional neural network (CNN). The project utilizes the PyTorch framework to gain insights into how these models function, tune their hyperparameters, and evaluate their performance.

Our CNN implementation focuses on the **LeNet-5 architecture** due to its historical significance and its foundational value in understanding convolutional neural networks.

---

## Project Structure

### Part 1: Perceptron
- **Objective**: Analyze and explain the tensor dimensions in `perceptron_pytorch.py`.
- **Details**:
  - Dataset tensors (`data_train`, `label_train`, etc.) for training and testing, including their shapes.
  - Model weight and bias initialization (`w`, `b`) and their shapes.
  - Training process including batch tensors and gradient updates.

### Part 2: Shallow Neural Network
- **Objective**: Implement and optimize a single-hidden-layer MLP.
- **Key Features**:
  - Split data into training, validation, and test sets to avoid overfitting.
  - Use **ReLU** activation and **Cross-Entropy Loss**.
  - Systematic hyperparameter tuning via grid search:
    - Learning rate: `[0.01, 0.001, 0.0001]`
    - Hidden neurons: `[64, 128, 256]`
    - Batch size: `[32, 64, 128]`
  - Best Parameters:
    - Learning rate: `0.001`
    - Hidden neurons: `256`
    - Batch size: `64`
  - Results:
    - Training accuracy: ~99%
    - Validation accuracy: ~97%

### Part 3: Deep Neural Network
- **Objective**: Extend the shallow network to include four hidden layers.
- **Key Features**:
  - Hierarchical feature learning with consistent hidden layer sizes.
  - Hyperparameter tuning similar to the shallow network:
    - Best parameters: Learning rate `0.001`, hidden neurons `256`, batch size `64`.
  - Results:
    - Training accuracy: ~99.5%
    - Validation accuracy: ~98%
  - Additional Insights:
    - Improved feature abstraction.
    - Stable training and gradient flow through multiple layers.

### Part 4: LeNet-5
- **Objective**: Implement LeNet-5 for robust image classification.
- **Architecture**:
  - Convolutional layers followed by average pooling and fully connected layers.
  - Use **Tanh** activation for all layers.
- **Key Features**:
  - Hyperparameter tuning:
    - Learning rate: `0.001`
    - Batch size: `64`
  - Results:
    - Validation accuracy: ~98.62%
    - External testing with real-world handwritten digits.
  - Practical Contributions:
    - Demonstrates CNN fundamentals and inference pipeline for real-world applications.

---


### Prerequisites
- Python 3.x
- PyTorch
- NumPy
- Matplotlib (optional, for visualizations)


