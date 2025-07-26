import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Assume these are already created in a previous notebook or script
# from your feature engineering code
from joblib import load

# Load preprocessed data (adjust if you saved them in files instead of memory)
X_train = load(r"C:\Users\pc\Fraud-Detection\notebooks\artifacts/X_train_bal.joblib")
y_train = load(r"C:\Users\pc\Fraud-Detection\notebooks\artifacts/y_train_bal.joblib")
X_test = load(r"C:\Users\pc\Fraud-Detection\notebooks\artifacts/X_test_final.joblib")
y_test = load(r"C:\Users\pc\Fraud-Detection\notebooks\artifacts/y_test.joblib")

# Step 1: Train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 2: Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Step 3: Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Step 4: Save the model
joblib.dump(model, "models/logistic_regression_model.joblib")
