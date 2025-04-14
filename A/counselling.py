# -*- coding: utf-8 -*-
# Counselling System - Prediction Model (Python Script Version)

import subprocess
import sys

# Ensure required packages are installed
required_packages = ["eli5", "shap", "lime"]
for package in required_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance

# Load data
df = pd.read_excel("Copy of cleaned_josaa_data_(1)(1).xlsx")  # Use correct path
threshold = 10000
df['admission_status'] = (df['closing_rank'] <= threshold).astype(int)
df.drop(columns=['year', 'round'], inplace=True)

# Encode categorical columns
categorical_columns = ['institute', 'academic_program_name', 'quota', 'seat_type', 'gender']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare X and y
X = df.drop(columns=['closing_rank', 'admission_status'])
y = df['admission_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
y_proba_lr = model_lr.predict_proba(X_test)[:, 1]

# Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
y_proba_rf = model_rf.predict_proba(X_test)[:, 1]

# Support Vector Classifier
model_svc = SVC(kernel='rbf', probability=True, random_state=42)
model_svc.fit(X_train, y_train)
y_pred_svc = model_svc.predict(X_test)
y_proba_svc = model_svc.predict_proba(X_test)[:, 1]

# Evaluation
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))
print("Random Forest:\n", classification_report(y_test, y_pred_rf))
print("SVC:\n", classification_report(y_test, y_pred_svc))

# Confusion Matrix for SVC
plt.figure(figsize=(6,6))
sns.heatmap(confusion_matrix(y_test, y_pred_svc), annot=True, fmt='d', cmap='Blues')
plt.title("SVC Confusion Matrix")
plt.show()

# ROC Curve
for model_name, fpr, tpr, roc_auc in [
    ("LR", *roc_curve(y_test, y_proba_lr)[:2], auc(*roc_curve(y_test, y_proba_lr)[:2])),
    ("RF", *roc_curve(y_test, y_proba_rf)[:2], auc(*roc_curve(y_test, y_proba_rf)[:2])),
    ("SVC", *roc_curve(y_test, y_proba_svc)[:2], auc(*roc_curve(y_test, y_proba_svc)[:2]))
]:
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(); plt.show()

# LIME Explainer
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=['Not Admitted', 'Admitted'],
    mode='classification'
)
exp = explainer.explain_instance(X_test.values[0], model_svc.predict_proba)
exp.show_in_notebook(show_table=True)  # Only works in notebooks

# Permutation Importance
perm = permutation_importance(model_svc, X_test, y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm.importances_mean
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance - SVM"); plt.tight_layout(); plt.show()

# Ensemble Prediction Function
def final_model_predict(X_input):
    proba_lr = model_lr.predict_proba(X_input)[:, 1]
    proba_rf = model_rf.predict_proba(X_input)[:, 1]
    proba_svc = model_svc.predict_proba(X_input)[:, 1]
    weighted_proba = 0.25 * proba_lr + 0.50 * proba_rf + 0.25 * proba_svc
    return (weighted_proba >= 0.5).astype(int), weighted_proba

# Manual Prediction Input
print("\n--- Custom Prediction ---")
input_data = {}
for col in ['institute', 'academic_program_name', 'quota', 'seat_type', 'gender']:
    raw = input(f"Enter {col.replace('_', ' ').title()}: ")
    input_data[col] = label_encoders[col].transform([raw])[0]

input_data['opening_rank'] = int(input("Enter Opening Rank: "))
input_df = pd.DataFrame([input_data])

predicted_class, probability = final_model_predict(input_df)
print("\n✅ Final Prediction:")
print("Admission Status (0 = Rejected, 1 = Admitted):", predicted_class[0])
print("Probability of Admission:", probability[0])