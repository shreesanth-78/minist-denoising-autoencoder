# MNIST Denoising Autoencoder

This repository contains a Deep Learning project focused on building a Denoising Autoencoder for the MNIST dataset. The project demonstrates the pipeline of artificially injecting Gaussian noise into images and training a neural network to reconstruct the original, clean digits.

## Project Structure

* `notebooks/Denoising_Autoencoder_MNIST_AID_B_Group14_152.ipynb`: The core project notebook. It contains the data loading pipeline, model architecture, training loops, visualization of reconstructed images, and a comparison of bottlenecks.
* `data/`: Directory for the downloaded MNIST dataset (ignored by Git).
* `models/`: Directory for saved PyTorch model weights (ignored by Git).
* `images/`: Directory for saving evaluation plots and output comparisons.
* `reports/`: Directory for CSV outputs and model parameter comparisons.

## Setup & Installation

1. Clone the repository.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt