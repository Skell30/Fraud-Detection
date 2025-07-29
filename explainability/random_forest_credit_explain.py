import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# STEP 1: Load model and data
print("Loading model and data...")
X_test = joblib.load("artifacts/X_credit_test.joblib")
model = joblib.load("artifacts/random_forest_credit_model.joblib")

# Convert numpy array to DataFrame if needed
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test)

# STEP 2: Create SHAP explainer
print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

# STEP 3: Calculate SHAP values (limit to 100 rows)
print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_test.iloc[:100], check_additivity=False)

# STEP 4: Determine correct SHAP value array for plotting
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # Class 1 (fraud)
else:
    shap_vals = shap_values

# STEP 5: Summary Plot
print("Generating summary plot...")
X_test.columns = X_test.columns.astype(str)  # <- 🔧 Fix column name types
shap.summary_plot(shap_vals, X_test.iloc[:100], show=False)
plt.tight_layout()
plt.savefig("outputs/credit_summary_plot.png")
plt.close()
print("✅ Saved summary plot to outputs/credit_summary_plot.png")


# STEP 6: Force Plot
print("Generating force plot...")
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
    shap_vals[0],
    X_test.iloc[0],
    matplotlib=True
)
plt.savefig("outputs/credit_force_plot.png")
plt.close()
print("✅ Saved force plot to outputs/credit_force_plot.png")



