# 🎓 Student Dropout Prediction Model

A machine learning project that predicts whether a student will **Graduate** or **Drop Out** using an ensemble of Random Forest and XGBoost classifiers — trained on academic, demographic, and socioeconomic data.

---

## 📌 Project Overview

Student dropout is a critical problem for educational institutions. This project builds a binary classification model to identify at-risk students early, enabling timely interventions. The model achieves high accuracy by combining two powerful algorithms in a soft-voting ensemble and engineering meaningful features from raw academic data.

**Target Classes:**
- `0` → Dropout
- `1` → Graduate

> ⚠️ Students with status `Enrolled` are excluded to keep the classification clean and maximize accuracy.

---

## 📁 Repository Structure

```
student-dropout-prediction/
│
├── data/
│   ├── dataset.csv          # Full training dataset
│   └── test.csv             # Test/inference dataset
│
├── saved_models/
│   └── ensemble_model.joblib  # Trained model (auto-generated)
│
├── model.py                 # Main training script
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 📊 Dataset

The dataset contains academic and personal records of higher education students. Each row represents one student.

| Feature Category | Examples |
|---|---|
| **Demographics** | Marital status, Age at enrollment, Gender, Nationality |
| **Application Info** | Application mode, Application order, Course |
| **Academic Performance** | Curricular units enrolled/approved/graded (Sem 1 & 2) |
| **Socioeconomic** | Debtor, Tuition fees up to date, Scholarship holder |
| **Macroeconomic** | Unemployment rate, Inflation rate, GDP |

**Target column:** `Target` — Graduate / Dropout (Enrolled rows dropped)

---

## 🔧 Feature Engineering

Three custom features are created to improve model performance:

| Feature | Formula | What it captures |
|---|---|---|
| `grade_momentum` | Sem 2 grade − Sem 1 grade | Whether a student is improving or declining |
| `eval_efficiency` | Sem 2 approved ÷ Sem 2 evaluations | How effectively a student converts attempts to passes |
| `socio_economic_risk` | Debtor + (1 − Tuition up to date) + Unemployment/20 | Combined financial & economic pressure |

---

## 🤖 Model Architecture

A **Soft Voting Ensemble** combining:

| Model | Config | Role |
|---|---|---|
| **Random Forest** | 300 trees, max depth 12 | Stable, interpretable baseline |
| **XGBoost** | 300 estimators, lr=0.05, depth 6 | Precise, gradient-boosted predictions |

The ensemble uses **soft voting** — both models output probabilities, which are averaged before the final class is decided. This reduces variance compared to using either model alone.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/student-dropout-prediction.git
cd student-dropout-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python model.py
```

This will:
- Load and preprocess `data/dataset.csv`
- Engineer new features
- Train the ensemble model
- Print accuracy and a full classification report
- Display a feature importance chart
- Save the trained model to `saved_models/ensemble_model.joblib`

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

---

## 📈 Results

After training, the model outputs:

```
✅ Final Model Accuracy: XX.XX%

Detailed Performance Report:
              precision    recall  f1-score   support
           0       ...       ...      ...       ...
           1       ...       ...      ...       ...
```

A **bar chart of the top 10 most important predictors** is also generated, giving interpretability to the model's decisions.

---

## 🔍 Key Predictors

Based on feature importance from the Random Forest component, the most influential factors in predicting dropout are typically:

1. 2nd semester approved units
2. 2nd semester grade
3. Curricular units grade momentum
4. Tuition fees up to date
5. Age at enrollment

---

## 🗺️ Roadmap

- [ ] Add SHAP values for per-student explainability
- [ ] Build a simple web UI for inference (Streamlit/Flask)
- [ ] Extend to multi-class (include Enrolled status with time-series approach)
- [ ] Hyperparameter tuning with Optuna or GridSearchCV
- [ ] Cross-validation for more robust performance estimates



#
