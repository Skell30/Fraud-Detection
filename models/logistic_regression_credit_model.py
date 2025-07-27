# models/logistic_regression_credit_model.py

import numpy as np
from joblib import load, dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load preprocessed data
X_train = load(r"C:\Users\pc\Fraud-Detection\artifacts/X_credit_train_bal.joblib")
y_train = load(r"C:\Users\pc\Fraud-Detection\artifacts/y_credit_train_bal.joblib")
X_test = load(r"C:\Users\pc\Fraud-Detection\artifacts/X_credit_test.joblib")
y_test = load(r"C:\Users\pc\Fraud-Detection\artifacts/y_credit_test.joblib")

# Train logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

roc_score = roc_auc_score(y_test, y_proba)
print(f"\nROC AUC Score: {roc_score:.4f}")

# Save model
dump(model, "artifacts/logistic_regression_credit_model.joblib")
