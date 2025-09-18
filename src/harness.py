import streamlit as st
import pandas as pd
import joblib

# Load models
rf_build_time = joblib.load("RF_build_time.pkl")
xgb_cost = joblib.load("XGB_material_cost.pkl")
clf_defect = joblib.load("clf_defect.pkl")

st.title("Harness Prediction")

# Manual numeric input
wire_count = st.number_input("Wire Count", min_value=1, value=50)
connector_count = st.number_input("Connector Count", min_value=0, value=10)
splice_count = st.number_input("Splice Count", min_value=0, value=2)
branch_points = st.number_input("Branch Points", min_value=0, value=3)
avg_wire_length_m = st.number_input("Average Wire Length (m)", min_value=0.1, value=1.5)
total_wire_length_m = st.number_input("Total Wire Length (m)", min_value=0.1, value=100.0)
dominant_awg = st.number_input("Dominant AWG", min_value=10, max_value=30, value=20)

if st.button("Predict"):
    # Create input dataframe
    df_input = pd.DataFrame({
        "wire_count":[wire_count],
        "connector_count":[connector_count],
        "splice_count":[splice_count],
        "branch_points":[branch_points],
        "avg_wire_length_m":[avg_wire_length_m],
        "total_wire_length_m":[total_wire_length_m],
        "dominant_awg":[dominant_awg]
    })
    
    # Align columns to model's expected features
    feature_names = rf_build_time.feature_names_in_
    X_prepared = df_input.reindex(columns=feature_names, fill_value=0)
    
    # Predict only if shapes match
    if X_prepared.shape[1] == len(feature_names):
        pred_build_time = rf_build_time.predict(X_prepared)[0]
        pred_cost = xgb_cost.predict(X_prepared)[0]
        pred_defect_prob = clf_defect.predict_proba(X_prepared)[:,1][0]
        pred_defect_label = "Yes" if pred_defect_prob > 0.5 else "No"
        
        st.subheader("Predictions")
        st.write(f"**Estimated Build Time (hr):** {pred_build_time:.2f}")
        st.write(f"**Estimated Material Cost (USD):** {pred_cost:.2f}")
        st.write(f"**Defect Probability:** {pred_defect_prob:.3f}")
        st.write(f"**Defect Likely:** {pred_defect_label}")
    else:
        st.error("Feature count mismatch! Check debug info above.")
