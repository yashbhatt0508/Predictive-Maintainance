# ⚙️ Machine Failure Prediction

An end-to-end Machine Learning project that predicts the probability of machine failure from operating and sensor conditions using an optimized **XGBoost Classifier**.

The project goes beyond model training by covering **data preprocessing, class imbalance, cross-validation, hyperparameter tuning, decision-threshold optimization, error analysis, SHAP explainability, and deployment with Streamlit.**

---

## 🚀 Live Demo

🔗 
https://predictive-maintainance-ml.streamlit.app/
---

## 📌 Project Overview

Unexpected machine failures can cause production downtime, maintenance costs, and equipment damage.

This project aims to predict whether a machine is likely to fail based on its operating conditions and sensor measurements.

The model uses:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The final application provides a **failure probability** and converts that probability into a failure/no-failure prediction using a selected decision threshold.

---

## 🧠 Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Preprocessing
   ↓
Train/Test Split
   ↓
Model Comparison
   ↓
Class Imbalance Analysis
   ↓
Scikit-learn Pipeline
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Final XGBoost Model
   ↓
Decision Threshold Optimization
   ↓
Error Analysis
   ↓
SHAP Explainability
   ↓
Model Serialization
   ↓
Streamlit Deployment



🎯 What I Learned

This project helped me understand the complete lifecycle of a machine learning classification project, including:

Performing meaningful EDA
Cleaning and preparing data
Handling class imbalance
Comparing classification models
Building preprocessing pipelines
Using cross-validation
Performing hyperparameter tuning
Understanding precision, recall, and F1
Optimizing classification thresholds
Investigating false positives and false negatives
Explaining model predictions with SHAP
Saving trained ML pipelines
Building an interactive ML application
Deploying a machine learning model with Streamlit
