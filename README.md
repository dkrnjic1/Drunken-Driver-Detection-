# Drunken Driver Detection

This project was developed as part of the "Artificial Intelligence" course at the Faculty of Electrical Engineering, University of Sarajevo. It explores the use of machine learning and computer vision techniques to detect drivers under the influence of alcohol using thermal images.

## 👥 Team Members

- Džana Krnjić
- Irma Galijašević 
- Ayša Nil

---

##  Problem Overview

Driving under the influence of alcohol is a major global safety concern, contributing significantly to traffic accidents. The goal of this project is to design and evaluate an AI system capable of identifying whether a driver is sober or drunk, using infrared facial images.

---

##  Key Concepts

- **Dataset**: Pre-labeled image datasets used for training and testing.
- **Classification**: Binary classification between "sober" and "drunk".
- **Computer Vision**: Detecting visual signs of intoxication on the face.
- **Deep Learning / CNN**: Used to learn image patterns automatically.

---

##  Dataset Details

- **Name**: [Drunk vs Sober Infrared Image Dataset](https://www.kaggle.com/datasets/kipshidze/drunk-vs-sober-infrared-image-dataset)
- **Source**: Kaggle 
- **Format**: Thermal infrared face images (.jpg)
- **Classes**:
  - `sober`
  - `20mins` (after alcohol intake)
  - `40mins`
  - `60mins`
- **Total samples**: 486
- **Preprocessing**:
  - Jet color mapping for visualization
  - Resizing to 256x256
  - Normalization to range [0, 1]

---

##  Model Architecture

A Convolutional Neural Network (CNN) was used, composed of:
- Multiple `Conv2D + MaxPooling` layers
- Flatten and Dense layers
- Sigmoid output activation (binary classification)

---

## 🛠 Technologies Used

- **Python**
- **TensorFlow / Keras**: Model training and evaluation
- **OpenCV**: Image conversion and color mapping
- **NumPy**: Data handling
- **scikit-learn**: Evaluation metrics
- **Matplotlib / Seaborn**: Visualization

---

## 🏁 Training & Evaluation

- **Optimizer**: Adam  
- **Loss Function**: Binary Crossentropy  
- **Epochs**: 30  
- **Validation split**: ~20%

### 📊 Performance Metrics on Test Set

| Metric       | Sober Class | Drunk Class |
|--------------|-------------|-------------|
| Precision    | 0.33        | 0.77        |
| Recall       | 0.17        | 0.89        |
| F1-score     | 0.22        | 0.83        |
| Accuracy     | 72%         |             |

> The model performed better in identifying drunk drivers, while sober driver detection suffered due to dataset imbalance.

---

##  Limitations and Improvements

- **Class imbalance** affected sober class detection.
- Using `class_weights` or **data augmentation** could help balance performance.
- A more powerful pre-trained model (e.g., MobileNet or EfficientNet) could improve accuracy.
- Dataset size and diversity could be improved for better generalization.

---

## 📎 Notebook & Code

All source code and experiments are available in the notebook:

[`DrunkenDriverDetection.ipynb`](./DrunkenDriverDetection.ipynb)
