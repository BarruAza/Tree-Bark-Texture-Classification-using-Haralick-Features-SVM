# Parallel Image Classification: Feature Extraction and SVM

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-red.svg)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-orange.svg)
![Multithreading](https://img.shields.io/badge/Parallelism-Multithreading-green.svg)

## Project Description

This project implements an image classification system using a traditional machine learning approach.  
The main focus is on efficient **feature extraction** from image datasets by applying **multithreading** using `concurrent.futures.ThreadPoolExecutor`.  
Extracted features, including **Haralick Texture** and **Color Histogram**, are then used to train a **Support Vector Machine (SVM)** model for image classification.

The dataset used in this project was obtained from the [Kaggle](https://www.kaggle.com/) website.  
Each image represents a bark texture sample from different tree species, making it suitable for texture-based classification tasks.

---

## Key Features

- **Parallel Feature Extraction:** Processes multiple images simultaneously using multithreading (default: 8 threads).  
- **Hybrid Feature Extraction:** Combines Haralick texture descriptors and color histogram features (8×8×8 bins).  
- **Image Enhancement:** Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve local contrast before feature extraction.  
- **Classification Model:** Implements a Support Vector Machine (SVM) with an RBF kernel inside a Scikit-learn `Pipeline`.  
- **Evaluation and Visualization:** Splits data into training and testing sets, computes accuracy, and saves visual prediction results.

---

## Dependencies

Before running the project, make sure the required Python libraries are installed:

```bash
pip install opencv-python mahotas numpy scikit-learn joblib
