# ⚡ Harness Manufacturing Prediction

This project predicts **material cost**, **build time**, and **defect probability** for harness (cable/wiring) manufacturing using machine learning.
By training separate models for each task, the system provides **early insights for production planning, cost optimization, and quality assurance**.

## 📌 Problem Statement

Harness manufacturing involves complex design and production steps.
Accurately estimating costs, time, and defect risks before production can:

* 💰 Reduce unexpected expenses
* 📅 Improve scheduling & resource allocation
* ✅ Minimize quality issues

## 📂 Project Structure

```
Capstone-Project1/
│
├── data/
│   └── synthetic_harness.csv   # Synthetic dataset of harness features
│
├── notebooks/
│   └── harness.ipynb           # Training, evaluation, visualization
│
├── src/
│   └── harness.py              # Script to load models & generate predictions
│
├── models/
│   └── README.md               # Placeholder 
│
└── README.md                   # Project documentation
```

## ⚙️ Workflow

**1. Data Preprocessing**

* Encode categorical variables
* Ensure consistent feature ordering
* Handle missing values / normalization

**2. Model Training**

* Material Cost → XGBoost Regressor
* Build Time → Random Forest Regressor + Linear Regression (baseline)
* Defect Prediction → Classifier

**3. Evaluation Metrics**

* Regression → MAE, RMSE, R²
* Classification → Accuracy, Precision, Recall, F1-Score, ROC-AUC


## 📊 Results (Validation Scores)

| Task              | Model                   | Metric   | Score |
| ----------------- | ----------------------- | -------- | ----- |
| Material Cost     | XGBoost Regressor       | R²       | 0.92  |
|                   |                         | RMSE     | 450.3 |
| Build Time        | Random Forest Regressor | R²       | 0.89  |
|                   | Linear Regression       | R²       | 0.72  |
| Defect Prediction | Classifier              | Accuracy | 0.87  |
|                   |                         | F1-Score | 0.84  |
|                   |                         | ROC-AUC  | 0.90  |


## 🚀 Business Impact

* **Cost Prediction** → Supports budgeting & supplier negotiations
* **Time Estimation** → Improves scheduling & workforce management
* **Defect Prediction** → Enhances quality control & reduces rework

## 🛠 Tech Stack

* Python (Pandas, NumPy, Scikit-learn, XGBoost)
* Jupyter Notebook
