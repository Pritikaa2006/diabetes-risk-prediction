# 🩺 Explainable Diabetes Risk Prediction System

An end-to-end machine learning application for estimating diabetes/prediabetes risk from clinical and lifestyle indicators, with **SHAP-based explainability** to show which factors contributed most to an individual prediction.

> **Note:** This project is an educational machine learning screening application and is not a medical diagnostic tool.

---

## 📌 Project Overview

Diabetes is a major public health concern, and identifying individuals who may be at higher risk can support early screening and awareness.

This project builds an **Explainable Machine Learning system** that:

- Processes clinical and lifestyle-related health indicators
- Trains and compares multiple machine learning models
- Handles class imbalance using a balanced XGBoost model
- Provides probability-based risk estimation
- Uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions
- Provides an interactive **Streamlit web application**

The goal is not only to make a prediction, but also to answer:

> **"Why did the model make this prediction?"**

---

## 🎯 Problem Statement

Traditional machine learning models can provide predictions without explaining the factors behind them.

For healthcare-related applications, interpretability is particularly important.

This project addresses this problem by developing an explainable diabetes risk prediction system that combines:

**Machine Learning + Class Imbalance Handling + SHAP Explainability + Interactive Web Application**

---

## 🚀 Key Features

### 1. Risk Prediction
Predicts the model-estimated probability of diabetes/prediabetes risk using clinical and lifestyle features.

### 2. Class Imbalance Handling
The dataset contains substantially more negative than positive samples.

A balanced XGBoost model was developed using `scale_pos_weight` to improve detection of the positive class.

### 3. Threshold-Based Screening
The application uses an evaluated operating threshold of **0.65** for its displayed screening category.

This threshold is a project-level model operating choice and is **not a medically validated clinical cutoff**.

### 4. Explainable AI
SHAP is used to explain:

- Global feature importance
- Individual predictions
- Features contributing toward higher predicted risk
- Features contributing toward lower predicted risk

### 5. Interactive Streamlit Application

Users can enter health-related information through a web interface and receive:

- Estimated risk probability
- Screening category
- SHAP explanation
- Feature contribution information

---

## 🧠 Machine Learning Workflow

```text
UCI Dataset
     ↓
Data Cleaning
     ↓
Duplicate Removal
     ↓
Exploratory Data Analysis
     ↓
Stratified Train/Test Split
     ↓
Model Training
     ↓
Model Comparison
     ↓
Balanced XGBoost
     ↓
Threshold Evaluation
     ↓
SHAP Explainability
     ↓
Streamlit Application