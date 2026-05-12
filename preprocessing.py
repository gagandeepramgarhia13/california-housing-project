# ============================================================
# STEP 2: Feature Engineering & Preprocessing
# California Housing Price Prediction Project
# ============================================================

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# ─────────────────────────────────────────
# 2.1  Reload dataset
# ─────────────────────────────────────────
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

print("=" * 55)
print("   STEP 2: FEATURE ENGINEERING & PREPROCESSING")
print("=" * 55)

# ─────────────────────────────────────────
# 2.2  Feature Engineering — new features
# ─────────────────────────────────────────
# Rooms per person
df['RoomsPerPerson']    = df['AveRooms']    / df['AveOccup']

# Bedroom ratio
df['BedroomRatio']      = df['AveBedrms']   / df['AveRooms']

# Income per occupant (proxy for wealth density)
df['IncomePerOccupant'] = df['MedInc']      / df['AveOccup']

print("\n  ✅  3 new features created:")
print("     • RoomsPerPerson    = AveRooms / AveOccup")
print("     • BedroomRatio      = AveBedrms / AveRooms")
print("     • IncomePerOccupant = MedInc / AveOccup")

print(f"\n  Dataset shape after engineering: {df.shape}")

# ─────────────────────────────────────────
# 2.3  Remove extreme outliers (IQR method)
#      Applied only to the target variable
# ─────────────────────────────────────────
Q1 = df['Price'].quantile(0.01)
Q3 = df['Price'].quantile(0.99)
before = len(df)
df = df[(df['Price'] >= Q1) & (df['Price'] <= Q3)]
print(f"\n  Outlier removal: {before - len(df)} rows removed from Price tails.")
print(f"  Remaining rows  : {len(df)}")

# ─────────────────────────────────────────
# 2.4  REGRESSION — Feature & Target split
# ─────────────────────────────────────────
FEATURE_COLS = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude',
    'RoomsPerPerson', 'BedroomRatio', 'IncomePerOccupant'
]

X = df[FEATURE_COLS]
y = df['Price']

# ─────────────────────────────────────────
# 2.5  Train / Test split
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n  Train set : {X_train.shape[0]} rows")
print(f"  Test  set : {X_test.shape[0]} rows")

# ─────────────────────────────────────────
# 2.6  Standard Scaling
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train ONLY
X_test_scaled  = scaler.transform(X_test)         # transform test

print("\n  ✅  StandardScaler applied.")
print(f"     Mean  (first feature after scaling): {X_train_scaled[:, 0].mean():.4f}")
print(f"     Std   (first feature after scaling): {X_train_scaled[:, 0].std():.4f}")

# ─────────────────────────────────────────
# 2.7  CLASSIFICATION target (for Logistic Regression)
# ─────────────────────────────────────────
median_price = df['Price'].median()
df['Price_Class'] = (df['Price'] > median_price).astype(int)

X_cls = df[FEATURE_COLS]
y_cls = df['Price_Class']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

scaler_cls = StandardScaler()
X_train_c_scaled = scaler_cls.fit_transform(X_train_c)
X_test_c_scaled  = scaler_cls.transform(X_test_c)

print(f"\n  Classification split (above/below median ${median_price*100:.0f}k):")
print(f"     Class 1 (above median): {y_cls.sum()} samples")
print(f"     Class 0 (below median): {(y_cls == 0).sum()} samples")

# ─────────────────────────────────────────
# 2.8  Save preprocessed data & scaler
# ─────────────────────────────────────────
import pickle, os
os.makedirs('models', exist_ok=True)

joblib.dump(scaler,     'models/scaler.pkl')
joblib.dump(scaler_cls, 'models/scaler_cls.pkl')

np.save('models/X_train.npy',   X_train_scaled)
np.save('models/X_test.npy',    X_test_scaled)
np.save('models/y_train.npy',   y_train.values)
np.save('models/y_test.npy',    y_test.values)

np.save('models/X_train_c.npy', X_train_c_scaled)
np.save('models/X_test_c.npy',  X_test_c_scaled)
np.save('models/y_train_c.npy', y_train_c.values)
np.save('models/y_test_c.npy',  y_test_c.values)

# Save feature column list
with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(FEATURE_COLS, f)

print("\n  ✅  Saved to models/:")
print("     scaler.pkl, scaler_cls.pkl")
print("     X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("     X_train_c.npy, X_test_c.npy, y_train_c.npy, y_test_c.npy")
print("     feature_cols.pkl")

print("\n\n✅  STEP 2 COMPLETE — Preprocessing finished.\n")