# 📂 Models

This folder stores trained models for harness manufacturing prediction.

Models are not included in this repository by default, you can **Train models locally** using `notebooks/harness.ipynb`

### Expected Models

* `clf_defect.pkl` → Classification model for defect prediction
* `XGB_material_cost.pkl` → XGBoost regressor for material cost
* `RF_build_time.pkl` → Random Forest regressor for build time
* `LR_build_time.pkl` → Linear Regression baseline model
* `encoder.pkl` → Pre-trained encoder for categorical features
* `feature_order.pkl` → Ensures consistent feature ordering
