# Harness Manufacturing Prediction

This project focuses on predicting **material cost**, **build time**, and **defect probability** for harness (cable/wiring) manufacturing using machine learning. By building separate models for each task, the project provides early insights into production planning, cost optimization, and quality assurance.

## 📌 Problem Statement

Harness manufacturing involves complex design and production steps. Accurately estimating **costs**, **manufacturing time**, and **defects** before production can:

* Reduce unexpected expenses
* Improve scheduling and resource allocation
* Minimize quality issues

## 📂 Project Structure

* `synthetic_harness.csv` → Synthetic dataset of harness features
* `harness.ipynb` → Jupyter notebook for training, evaluation, and visualization
* `harness.py` → Script to load models and generate predictions
* `encoder.pkl` → Pre-trained encoder for categorical features
* `feature_order.pkl` → Ensures correct feature alignment during inference
* `clf_defect.pkl` → Classification model for defect prediction
* `XGB_material_cost.pkl` → XGBoost regression model for material cost prediction
* `RF_build_time.pkl` → Random Forest regression model for build time estimation
* `LR_build_time.pkl` → Linear Regression baseline model for build time

## ⚙️ Workflow

**Steps:**

1. **Data Preprocessing**

   * Encoding categorical variables
   * Ensuring consistent feature ordering
   * Normalization & handling missing values

2. **Model Training**

   * **Material Cost** → XGBoost Regressor
   * **Build Time** → Random Forest Regressor & Linear Regression (baseline)
   * **Defect Prediction** → Classifier

3. **Evaluation Metrics**

   * Regression → MAE, RMSE, R²
   * Classification → Accuracy, Precision, Recall, F1-score, ROC-AUC

4. **Deployment-Ready Models**

   * Models saved as `.pkl` files for direct use in production scripts.


## 📊 Results

| Task              | Model                   | Metric   | Score  |
| ----------------- | ----------------------- | -------- | ------ |
| Material Cost     | XGBoost Regressor       | R²       | 0.92   |
|                   |                         | RMSE     | 450.32 |
| Build Time        | Random Forest Regressor | R²       | 0.89   |
|                   | Linear Regression       | R²       | 0.72   |
| Defect Prediction | Classifier              | Accuracy | 0.87   |
|                   |                         | F1-Score | 0.84   |
|                   |                         | ROC-AUC  | 0.90   |

*(Scores are from validation runs in `harness.ipynb`.)*

## 🚀 Business Impact

* **Cost Prediction** → Supports budgeting and supplier negotiations
* **Time Estimation** → Improves scheduling & workforce management
* **Defect Prediction** → Enhances quality control and reduces rework

## 🛠 Tech Stack

* Python (Pandas, NumPy, Scikit-learn, XGBoost)
* Jupyter Notebook

