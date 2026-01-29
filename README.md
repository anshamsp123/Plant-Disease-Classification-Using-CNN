# Plant Disease Classification Using Convolutional Neural Networks (CNN)

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)**–based image classification system to identify **plant leaf diseases** using the **PlantVillage dataset**. The model is trained using **TensorFlow and Keras** and is capable of classifying plant leaf images into multiple disease categories with high accuracy.

The project was developed as part of **Deep Learning Lab – Assignment 03** and demonstrates the complete deep learning workflow including dataset loading, preprocessing, model design, training, evaluation, and prediction on custom images.

---

## 🎯 Objectives  
Load and preprocess image datasets using TensorFlow Datasets  
Design a CNN architecture for multi-class classification  
Train and validate the model on plant disease images  
Evaluate performance using accuracy and loss metrics  
Predict disease class for unseen plant images

---

## Dataset Used
**PlantVillage Dataset**  
Contains images of healthy and diseased plant leaves  
Covers **38 different classes**  
Publicly available via TensorFlow Datasets

Dataset Source:
tensorflow_datasets : plant_village

---

## ⚙️ Technologies Used  
Python  
TensorFlow  
Keras  
TensorFlow Datasets (TFDS)  
NumPy  
Matplotlib  
Google Colab

---

## Data Preprocessing
- Images resized to **128 × 128**
- Pixel values normalized to range **[0, 1]**
- Dataset split into:
  - Training set (20%)
  - Validation/Test set (5%)
- Batched and prefetched for performance optimization

---

## 🏗️ Model Architecture
The CNN architecture consists of:

| Layer | Description |
|------|------------|
| Conv2D | 32 filters, ReLU activation |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, ReLU activation |
| MaxPooling2D | 2 × 2 |
| Flatten | Converts 2D features to 1D |
| Dense | 128 neurons, ReLU |
| Dropout | 0.5 |
| Output Dense | Softmax (38 classes) |

**Loss Function:** Sparse Categorical Crossentropy  
**Optimizer:** Adam  

---

## Training Details  
**Epochs:** 10  
**Batch Size:** 32  
**Final Validation Accuracy:** ~81%  
**Final Test Accuracy:** ~81.5%

---

## Performance Visualization
The model training process includes:
- Training vs Validation Accuracy Graph
- Training vs Validation Loss Graph

These plots help analyze convergence and overfitting behavior.

---

## Model Evaluation
```text
Test Accuracy: 0.820
