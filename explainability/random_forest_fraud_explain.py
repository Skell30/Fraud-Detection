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

shap_values = explainer(X_test.iloc[:100])

print("Plotting summary plot...")

# Summary Plot (Global importance)
shap.summary_plot(shap_values[1], X_test, show=False)
print("Done.")
plt.tight_layout()
plt.savefig(r"C:\Users\pc\Fraud-Detection\outputs/fraud_summary_plot.png")
plt.close()

# Force Plot (Local explanation for 1st prediction)
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0], matplotlib=True)
plt.savefig(r"C:\Users\pc\Fraud-Detection\outputs/fraud_force_plot.png")
