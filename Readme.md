# 🏠 California Housing Price Prediction System

End-to-end Machine Learning + Deep Learning project using the
California Housing dataset from scikit-learn.

---

## 📁 Project Structure

```
CaliforniaHousingProject/
│
├── step1_eda.py              ← EDA & visualizations
├── step2_preprocessing.py   ← Feature engineering & scaling
├── step3_ml_models.py       ← ML models + hyperparameter tuning
├── step4_deep_learning.py   ← TensorFlow neural network
├── app.py                   ← Streamlit web application
├── requirements.txt
├── README.md
│
└── models/                  ← Created automatically after running steps
    ├── scaler.pkl
    ├── scaler_cls.pkl
    ├── feature_cols.pkl
    ├── linear_regression.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── deep_learning_model.keras
    └── deep_learning_best.keras
```

---

## ⚙️ Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Run Order (IMPORTANT)

Run the steps **in order**:

```bash
# Step 1 — Exploratory Data Analysis
python step1_eda.py

# Step 2 — Preprocessing (creates models/ folder with .npy files)
python step2_preprocessing.py

# Step 3 — Train ML models (Linear, Decision Tree, Random Forest, XGBoost)
python step3_ml_models.py

# Step 4 — Train Deep Learning model
python step4_deep_learning.py

# Step 5 — Launch Streamlit dashboard
streamlit run app.py
```

---

## 🧠 Algorithms

| Model               | Type           | Tuning        |
|---------------------|----------------|---------------|
| Linear Regression   | Regression     | —             |
| Logistic Regression | Classification | —             |
| Decision Tree       | Regression     | GridSearchCV  |
| Random Forest       | Regression     | GridSearchCV  |
| XGBoost             | Regression     | GridSearchCV  |
| Neural Network      | Regression     | EarlyStopping |

---

## 📊 Expected Results

| Model             | Approx R² |
|-------------------|-----------|
| Linear Regression | ~0.60     |
| Decision Tree     | ~0.72     |
| Random Forest     | ~0.80     |
| XGBoost           | ~0.83     |
| Deep Learning     | ~0.85     |

---

## 🌐 Streamlit Dashboard Sections

| Section             | Description                                      |
|---------------------|--------------------------------------------------|
| 🏠 Home             | Overview, KPIs, raw data preview                 |
| 📊 Data Insights    | Statistics, correlation heatmap, distributions   |
| 📈 Visualizations   | Scatter, geo map, box plots, pair analysis       |
| 🤖 Predict Price    | Interactive prediction form with model selector  |
| 🏆 Model Comparison | Side-by-side metrics, actual vs predicted charts |

---

## 🔑 Key Features

- **Feature Engineering**: 3 new derived features (RoomsPerPerson, BedroomRatio, IncomePerOccupant)
- **Outlier Removal**: 1st–99th percentile clipping on target
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Deep Learning**: 4-layer network with BatchNormalization, Dropout, EarlyStopping
- **Streamlit**: Professional 5-page dashboard with Plotly charts and geo map

---

## 💡 Interview Q&A Quick Reference

**Why XGBoost?**
Gradient boosting builds trees sequentially — each corrects the previous one's errors — giving high accuracy with built-in regularization to prevent overfitting.

**Why feature scaling?**
Algorithms like Linear/Logistic Regression and Neural Networks are sensitive to feature magnitude. StandardScaler ensures all features contribute equally.

**What is hyperparameter tuning?**
Finding the best model configuration (e.g. max_depth, n_estimators) through systematic search (GridSearchCV) with cross-validation.

**What is overfitting?**
Model memorizes training data and fails to generalize. Mitigated by: Dropout (DL), min_samples_leaf (trees), regularization (XGBoost), EarlyStopping.

**Why Streamlit?**
Pure Python, minimal boilerplate, hot-reload, and native integration with pandas/plotly — ideal for rapid ML dashboard deployment.