from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from joblib import load, dump

# Load preprocessed and balanced training/testing sets
X_train = load(r"C:\Users\pc\Fraud-Detection\artifacts/X_train_bal.joblib")
X_test = load(r"C:\Users\pc\Fraud-Detection\artifacts/X_test_final.joblib")
y_train = load(r"C:\Users\pc\Fraud-Detection\artifacts/y_train_bal.joblib")
y_test = load(r"C:\Users\pc\Fraud-Detection\artifacts/y_test.joblib")

# Train Random Forest
model = RandomForestClassifier(random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Save model
dump(model, "artifacts/random_forest_model.joblib")
