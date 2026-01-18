# ============================================
# TASK 4: ADVANCED MACHINE LEARNING MODEL
# Student Performance Prediction (PASS / FAIL)
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# ============================================
# STEP 1: CREATE DATASET (NO FILE NEEDED)
# ============================================

data = {
    "Study_Hours": [1,2,3,4,5,6,2,3,4,5,6,1,2,4,6],
    "Attendance": [40,50,60,65,70,80,55,60,68,75,85,45,52,72,90],
    "Previous_Score": [35,45,55,60,65,75,50,58,62,70,80,40,48,68,85],
    "Result": ["Fail","Fail","Fail","Pass","Pass","Pass",
               "Fail","Fail","Pass","Pass","Pass","Fail","Fail","Pass","Pass"]
}

df = pd.DataFrame(data)

print("Dataset Preview:\n")
display(df.head())

# ============================================
# STEP 2: SPLIT FEATURES & LABEL
# ============================================

X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Result"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # Pass=1, Fail=0

# ============================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42
)

# ============================================
# STEP 4: PIPELINE (ADVANCED)
# ============================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", DecisionTreeClassifier(random_state=42))
])

# ============================================
# STEP 5: HYPERPARAMETER TUNING
# ============================================

params = {
    "model__max_depth": [2, 3, 4, 5],
    "model__min_samples_split": [2, 4, 6]
}

grid = GridSearchCV(
    pipeline,
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

# ============================================
# STEP 6: EVALUATION
# ============================================

best_model = grid.best_estimator_
predictions = best_model.predict(X_test)

print("\nBest Parameters:", grid.best_params_)
print("\nModel Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:\n")
print(classification_report(y_test, predictions, target_names=encoder.classes_))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, predictions))

# ============================================
# STEP 7: USER PREDICTION (INTERACTIVE)
# ============================================

def predict_student(study_hours, attendance, previous_score):
    sample = pd.DataFrame([[study_hours, attendance, previous_score]],
                          columns=X.columns)
    result = best_model.predict(sample)
    return encoder.inverse_transform(result)[0]

print("\n--- SAMPLE PREDICTIONS ---")
print("Student 1:", predict_student(5, 80, 75))
print("Student 2:", predict_student(2, 50, 45))
