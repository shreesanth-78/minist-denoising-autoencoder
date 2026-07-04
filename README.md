# MNIST Denoising Autoencoder & Latent Space Analysis

This repository contains a Deep Learning pipeline focused on building a Denoising Autoencoder (DAE) for the MNIST dataset. Beyond simple image reconstruction, this project explores the mathematical and practical implications of the **bottleneck** (latent space) by evaluating its effectiveness as a feature extractor for downstream classification tasks.

## 🚀 Project Overview

The primary objective is to train a neural network to reconstruct clean handwritten digits from inputs corrupted by Gaussian noise. Once the autoencoder is trained, the project extracts the encoder component to evaluate how well the compressed representations (bottleneck features) perform when passed into a secondary Convolutional Neural Network (CNN) classifier. 

By comparing different bottleneck dimensions, the project highlights the trade-off between dimensionality reduction, parameter efficiency, and classification accuracy.

## 🧠 Key Concepts Covered

* **Denoising Autoencoders (DAE):** Training a model to ignore "noise" by setting the target output to the original, uncorrupted input. This forces the model to learn the most robust and essential features of the data.
* **Gaussian Noise Injection:** Artificially corrupting the MNIST dataset to simulate real-world sensor noise or data transmission errors.
* **Latent Space & Dimensionality Reduction:** Compressing $28 \times 28$ (784 pixel) images into dense, lower-dimensional vectors. 
* **Feature Extraction:** Using a pre-trained encoder as a standalone feature extractor for a downstream machine learning task.
* **Model Parameter Efficiency:** Analyzing how the size of the input affects the total trainable parameters of downstream models.

## ⚙️ Methodology & Pipelines

The project evaluates two distinct classification pipelines to understand the value of the autoencoder's latent space:

**Pipeline 1: CNN on Corrupted Images (Baseline)**
* A standard CNN is trained directly on the noisy MNIST images.
* **Result:** High accuracy, but requires a large number of trainable parameters (~421,642) as the CNN must process the full $28 \times 28$ spatial dimensions.

**Pipeline 2: Classification at the Bottleneck**
* The noisy images are passed through the pre-trained Autoencoder's Encoder layer.
* The resulting compressed 1D tensor (the bottleneck) is passed to a smaller, dense classifier.
* **Result:** Highly parameter-efficient (e.g., ~4,810 parameters for a bottleneck of 64). 

## 📊 Results & Bottleneck Analysis

A core component of this project is testing how the size of the latent space impacts classification accuracy. The following bottleneck sizes were evaluated:

| Bottleneck Size | Accuracy (%) | Classifier Parameters |
|:---:|:---:|:---:|
| **128** | ~ 77.35% | 8,906 |
| **64** | ~ 70.86% | 4,810 |
| **32** | ~ 63.44% | 2,762 |
| **16** | ~ 59.81% | 1,738 |

**Conclusion:** A bottleneck size of **128** provided the optimal balance, preserving enough variance in the feature space to maintain strong classification accuracy while drastically reducing the parameter count compared to the baseline CNN.

## 💡 Real-World Applications

The concepts demonstrated in this project extend far beyond the MNIST dataset into industrial and research applications:
1. **Medical Imaging:** Removing artifacts and sensor noise from MRI or X-ray scans without losing critical anatomical details.
2. **Edge AI & IoT:** Compressing rich sensor data (like high-res images) on low-power edge devices before transmitting the lightweight latent vector to a central server for classification.
3. **Data Preprocessing:** Serving as a powerful, non-linear alternative to PCA (Principal Component Analysis) for dimensionality reduction prior to training traditional machine learning algorithms.
4. **Anomaly Detection:** Denoising Autoencoders naturally struggle to reconstruct data they haven't seen before. High reconstruction loss can be used as a trigger to flag defective products on an assembly line.

## 📁 Repository Structure

* `notebooks/` - Contains the core `.ipynb` Colab notebook with the model architectures and training loops.
* `data/` - Directory for the dataset (excluded from version control).
* `models/` - Directory for saved `.pth` PyTorch model weights (excluded from version control).
* `images/` - Visualizations of the loss curves, clean vs. noisy vs. reconstructed image comparisons, and confusion matrices.
* `reports/` - Exported `.csv` files detailing model parameter comparisons and bottleneck accuracy metrics.

## 🛠️ Setup & Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/shreesanth-78/minist-denoising-autoencoder.git](https://github.com/shreesanth-78/minist-denoising-autoencoder.git)
   cd minist-denoising-autoencoder