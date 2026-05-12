# ============================================================
# STEP 3: Machine Learning Models + Hyperparameter Tuning
# California Housing Price Prediction Project
# ============================================================
# Run AFTER step2_preprocessing.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, pickle, time, os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model   import LinearRegression, LogisticRegression
from sklearn.tree           import DecisionTreeRegressor
from sklearn.ensemble       import RandomForestRegressor
from xgboost                import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics        import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────
# 3.0  Load preprocessed data
# ─────────────────────────────────────────
print("=" * 55)
print("   STEP 3: MACHINE LEARNING MODELS")
print("=" * 55)

X_train = np.load('models/X_train.npy')
X_test  = np.load('models/X_test.npy')
y_train = np.load('models/y_train.npy')
y_test  = np.load('models/y_test.npy')

X_train_c = np.load('models/X_train_c.npy')
X_test_c  = np.load('models/X_test_c.npy')
y_train_c = np.load('models/y_train_c.npy')
y_test_c  = np.load('models/y_test_c.npy')

print(f"\n  Loaded: X_train {X_train.shape}, X_test {X_test.shape}")

results = {}   # store metrics for all models

# ─────────────────────────────────────────────────────────
# 3.1  LINEAR REGRESSION
# ─────────────────────────────────────────────────────────
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  3.1  LINEAR REGRESSION")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

mae_lr  = mean_absolute_error(y_test, pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
r2_lr   = r2_score(y_test, pred_lr)

print(f"  MAE       : {mae_lr:.4f}")
print(f"  RMSE      : {rmse_lr:.4f}")
print(f"  R² Score  : {r2_lr:.4f}")

results['Linear Regression'] = {'MAE': mae_lr, 'RMSE': rmse_lr, 'R2': r2_lr}
joblib.dump(lr, 'models/linear_regression.pkl')
print("  ✅ Model saved: models/linear_regression.pkl")

# ─────────────────────────────────────────────────────────
# 3.2  LOGISTIC REGRESSION (Classification)
# ─────────────────────────────────────────────────────────
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  3.2  LOGISTIC REGRESSION (Above / Below Median Price)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_c, y_train_c)
pred_cls = log_model.predict(X_test_c)

acc = accuracy_score(y_test_c, pred_cls)
print(f"  Accuracy  : {acc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test_c, pred_cls, target_names=['Below Median', 'Above Median']))

# Confusion matrix plot
cm = confusion_matrix(y_test_c, pred_cls)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ['Below Median', 'Above Median'])
plt.yticks([0, 1], ['Below Median', 'Above Median'])
plt.xlabel('Predicted', fontsize=10)
plt.ylabel('Actual', fontsize=10)
plt.title('Logistic Regression — Confusion Matrix', fontsize=11, fontweight='bold')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=14, color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig('logistic_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("  📊 Confusion matrix saved as logistic_confusion_matrix.png")

joblib.dump(log_model, 'models/logistic_regression.pkl')
print("  ✅ Model saved: models/logistic_regression.pkl")

# ─────────────────────────────────────────────────────────
# 3.3  DECISION TREE + GridSearchCV
# ─────────────────────────────────────────────────────────
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  3.3  DECISION TREE (with Hyperparameter Tuning)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

dt = DecisionTreeRegressor(random_state=42)
params_dt = {
    'max_depth'       : [3, 5, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4]
}

print("  Running GridSearchCV (cv=5) — please wait...")
t0 = time.time()
grid_dt = GridSearchCV(dt, params_dt, cv=5, scoring='r2', n_jobs=-1, verbose=0)
grid_dt.fit(X_train, y_train)
print(f"  Finished in {time.time()-t0:.1f}s")
print(f"  Best params: {grid_dt.best_params_}")
print(f"  Best CV R²  : {grid_dt.best_score_:.4f}")

best_dt  = grid_dt.best_estimator_
pred_dt  = best_dt.predict(X_test)

mae_dt   = mean_absolute_error(y_test, pred_dt)
rmse_dt  = np.sqrt(mean_squared_error(y_test, pred_dt))
r2_dt    = r2_score(y_test, pred_dt)

print(f"\n  MAE       : {mae_dt:.4f}")
print(f"  RMSE      : {rmse_dt:.4f}")
print(f"  R² Score  : {r2_dt:.4f}")

results['Decision Tree'] = {'MAE': mae_dt, 'RMSE': rmse_dt, 'R2': r2_dt}
joblib.dump(best_dt, 'models/decision_tree.pkl')
print("  ✅ Model saved: models/decision_tree.pkl")

# ─────────────────────────────────────────────────────────
# 3.4  RANDOM FOREST + GridSearchCV
# ─────────────────────────────────────────────────────────
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  3.4  RANDOM FOREST (with Hyperparameter Tuning)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

rf = RandomForestRegressor(random_state=42)
params_rf = {
    'n_estimators'    : [50, 100, 150],
    'max_depth'       : [5, 10, None],
    'min_samples_split': [2, 5],
}

print("  Running GridSearchCV (cv=3) — this may take ~2 min...")
t0 = time.time()
grid_rf = GridSearchCV(rf, params_rf, cv=3, scoring='r2', n_jobs=-1, verbose=0)
grid_rf.fit(X_train, y_train)
print(f"  Finished in {time.time()-t0:.1f}s")
print(f"  Best params: {grid_rf.best_params_}")
print(f"  Best CV R²  : {grid_rf.best_score_:.4f}")

best_rf = grid_rf.best_estimator_
pred_rf = best_rf.predict(X_test)

mae_rf  = mean_absolute_error(y_test, pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
r2_rf   = r2_score(y_test, pred_rf)

print(f"\n  MAE       : {mae_rf:.4f}")
print(f"  RMSE      : {rmse_rf:.4f}")
print(f"  R² Score  : {r2_rf:.4f}")

results['Random Forest'] = {'MAE': mae_rf, 'RMSE': rmse_rf, 'R2': r2_rf}
joblib.dump(best_rf, 'models/random_forest.pkl')
print("  ✅ Model saved: models/random_forest.pkl")

# ─────────────────────────────────────────────────────────
# 3.5  XGBOOST + GridSearchCV
# ─────────────────────────────────────────────────────────
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  3.5  XGBOOST (with Hyperparameter Tuning)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

xgb = XGBRegressor(random_state=42, verbosity=0)
params_xgb = {
    'n_estimators'  : [100, 200],
    'learning_rate' : [0.05, 0.1, 0.2],
    'max_depth'     : [3, 5, 7],
    'subsample'     : [0.8, 1.0]
}

print("  Running GridSearchCV (cv=3) — please wait...")
t0 = time.time()
grid_xgb = GridSearchCV(xgb, params_xgb, cv=3, scoring='r2', n_jobs=-1, verbose=0)
grid_xgb.fit(X_train, y_train)
print(f"  Finished in {time.time()-t0:.1f}s")
print(f"  Best params: {grid_xgb.best_params_}")
print(f"  Best CV R²  : {grid_xgb.best_score_:.4f}")

best_xgb = grid_xgb.best_estimator_
pred_xgb = best_xgb.predict(X_test)

mae_xgb  = mean_absolute_error(y_test, pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
r2_xgb   = r2_score(y_test, pred_xgb)

print(f"\n  MAE       : {mae_xgb:.4f}")
print(f"  RMSE      : {rmse_xgb:.4f}")
print(f"  R² Score  : {r2_xgb:.4f}")

results['XGBoost'] = {'MAE': mae_xgb, 'RMSE': rmse_xgb, 'R2': r2_xgb}
joblib.dump(best_xgb, 'models/xgboost.pkl')
print("  ✅ Model saved: models/xgboost.pkl  ← PRIMARY PREDICTION MODEL")

# ─────────────────────────────────────────────────────────
# 3.6  MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────
print("\n\n" + "=" * 55)
print("  MODEL COMPARISON SUMMARY")
print("=" * 55)

comparison_df = pd.DataFrame(results).T.round(4)
comparison_df.index.name = 'Model'
print(comparison_df.to_string())

# Bar chart — R² scores
plt.figure(figsize=(9, 5))
models_names = list(results.keys())
r2_scores    = [results[m]['R2'] for m in models_names]
colors = ['#B4B2A9', '#85B7EB', '#5DCAA5', '#EF9F27']
bars = plt.bar(models_names, r2_scores, color=colors, edgecolor='white', linewidth=0.8)
plt.ylim(0, 1)
plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='R²=0.8 baseline')
for bar, val in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.ylabel('R² Score', fontsize=11)
plt.title('Model Comparison — R² Score', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison_r2.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  📊 Comparison chart saved as model_comparison_r2.png")

# Feature Importance — XGBoost
with open('models/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

importances = best_xgb.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
feat_imp.plot(kind='barh', color='#EF9F27', edgecolor='white')
plt.title('XGBoost Feature Importance', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("  📊 Feature importance saved as feature_importance.png")

print("\n\n✅  STEP 3 COMPLETE — All ML models trained & saved.\n")