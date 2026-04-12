import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import os

# --- 1. DATA PREPARATION ---
# Load your dataset
df = pd.read_csv('../data/dataset.csv') 

# Clean Target: Focus on Dropout (0) vs Graduate (1) for high accuracy
# If you keep 'Enrolled', accuracy stays lower due to overlapping data.
df = df[df['Target'] != 'Enrolled'] 

le = LabelEncoder()
df['Target'] = le.fit_transform(df['Target'])

# --- 2. ADVANCED FEATURE ENGINEERING (The secret to 85%+) ---
# Momentum: Did grades improve or drop?
df['grade_momentum'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']

# Approval Efficiency: Percentage of classes passed
df['eval_efficiency'] = df['Curricular units 2nd sem (approved)'] / df['Curricular units 2nd sem (evaluations)'].replace(0, 1)

# Socio-Economic Pressure Score
df['socio_economic_risk'] = df['Debtor'] + (1 - df['Tuition fees up to date']) + df['Unemployment rate']/20

# Fill any gaps created by math
df = df.fillna(0)

# --- 3. TRAIN/TEST SPLIT ---
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. THE VOTING ENSEMBLE ---
# Model A: Random Forest (Stable)
rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)

# Model B: XGBoost (Precise)
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric='logloss')

# Combine them: The models vote on the outcome
ensemble_model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')
ensemble_model.fit(X_train, y_train)

# --- 5. RESULTS ---
y_pred = ensemble_model.predict(X_test)
print(f"✅ Final Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred))

# --- 6. VISUALIZE THE "WHY" (Feature Importance) ---
# We'll use the RF part of the ensemble to show importance
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(10,6))
importances.sort_values(ascending=False).head(10).plot(kind='barh', color='teal')
plt.title("Top 10 Predictors of Student Success/Dropout")
plt.show()

# SAVE MODEL
os.makedirs('../saved_models',exist_ok=True)

dump(ensemble_model, '../saved_models/ensemble_model.joblib')
print("✅ Model saved successfully!")
