# Vision Transformer on CIFAR-10

## Project Overview

This repository contains an implementation of an Encoder-only Vision Transformer (ViT) trained on the CIFAR-10 dataset. The project demonstrates how to build patch embeddings, add positional encodings, apply multi-head self-attention layers, and perform image classification using the CLS token representation.

## Repository Contents

* `Vision Transformer.ipynb`: Jupyter notebook with code for data loading, model implementation, training, evaluation, and attention visualization.
* `README.md`: Project documentation (this file).

## Notebook Structure

The `Vision Transformer.ipynb` notebook is organized into the following sections:

1. **Data Preparation**: Loading and normalizing CIFAR-10 images, creating train/test splits.
2. **Model Components**:

   * **Patch Embedding**: Splitting images into fixed-size patches and projecting them into embedding vectors.
   * **Positional Embeddings**: Adding learnable position encodings to patch embeddings.
   * **Transformer Encoder**: Stacking multiple multi-head self-attention and feed-forward layers.
   * **Classification Head**: Using the CLS token representation for final classification.
3. **Training Procedure**: Defining optimizer, loss function, and training loop; tracking accuracy and loss metrics.
4. **Performance Analysis**: Plotting training and validation loss/accuracy curves to assess learning behavior.
5. **Attention Visualization**:

   * Extracting attention weights from different layers and heads.
   * Visualizing spatial attention maps over input patches.
   * Interpreting how the model attends to global context versus local features.
6. **Hyperparameter Experiments**: Exploring the impact of embedding dimension, number of heads, number of encoder layers, and learning rate on model performance.

## Hyperparameter Configuration

Within the notebook, you can modify key hyperparameters:

* **Embedding Dimension** (e.g., 64, 128).
* **Number of Attention Heads** (e.g., 4, 8).
* **Number of Encoder Layers** (e.g., 6, 12).
* **Patch Size** (e.g., 4×4, 8×8).
* **Learning Rate** and **Optimizer Settings**.
* **Batch Size** and **Epochs**.

## Requirements

* Python 3.7+
* PyTorch
* torchvision
* matplotlib
* seaborn
* numpy
* tqdm

## Results Summary

* **Accuracy**: Achieved up to XX% test accuracy on CIFAR-10 with optimal hyperparameters.
* **Training Behavior**: Convergence observed within YY epochs; minimal overfitting with proper regularization.
* **Attention Insights**: Early layers focus on local textures; deeper layers capture global object structure.

## License

This project is licensed under the MIT License. Feel free to use and modify!
