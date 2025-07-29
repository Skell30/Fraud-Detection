import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load model and data
print("Loading model and data...")

model = joblib.load(r"C:\Users\pc\Fraud-Detection\artifacts/random_forest_model.joblib")
X_test = joblib.load(r"C:\Users\pc\Fraud-Detection\artifacts/X_test_final.joblib")

# Initialize SHAP explainer
print("Creating SHAP explainer...")

explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")

shap_values = explainer(X_test.iloc[:100], check_additivity=False)


print("Plotting summary plot...")

# Summary Plot (Global importance)
shap.summary_plot(shap_values, X_test[:100], show=False)
plt.tight_layout()
plt.savefig(r"C:\Users\pc\Fraud-Detection\outputs/fraud_summary_plot.png")
plt.close()
print("Done.")
# Force Plot (Local explanation for 1st prediction)
print("Plotting force plot...")

# Get SHAP explanation object for one sample
sample_shap = shap_values[0]

# Plot and save
shap.plots.force(
    sample_shap.base_values,
    sample_shap.values,
    sample_shap.data,
    matplotlib=True,
    show=False
)

plt.savefig(r"C:\Users\pc\Fraud-Detection\outputs/fraud_force_plot.png")
plt.close()

print("Force plot saved.")

