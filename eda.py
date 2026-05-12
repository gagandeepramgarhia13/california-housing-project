# ============================================================
# STEP 1: Import Libraries & Load Dataset
# California Housing Price Prediction Project
# ============================================================
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1.1  Load the dataset
# ─────────────────────────────────────────
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target          # Price is in units of $100,000

print("=" * 55)
print("   CALIFORNIA HOUSING DATASET - OVERVIEW")
print("=" * 55)
print(f"\n  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Features:\n  {list(df.columns)}")

# ─────────────────────────────────────────
# 1.2  First look at the data
# ─────────────────────────────────────────
print("\n\n--- First 5 Rows ---")
print(df.head())

print("\n\n--- Data Types ---")
print(df.dtypes)

# ─────────────────────────────────────────
# 1.3  Data Cleaning — check for nulls
# ─────────────────────────────────────────
print("\n\n--- Missing Values ---")
print(df.isnull().sum())
print("\n  ✅  No missing values in this dataset.")

# ─────────────────────────────────────────
# 1.4  Statistical Summary
# ─────────────────────────────────────────
print("\n\n--- Statistical Summary ---")
print(df.describe().round(2))

# ─────────────────────────────────────────
# 1.5  Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('California Housing — Feature Distributions', fontsize=16, fontweight='bold', y=1.01)

for i, col in enumerate(df.columns):
    ax = axes[i // 3][i % 3]
    df[col].hist(ax=ax, bins=40, color='#4C72B0', edgecolor='white', alpha=0.85)
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  📊  Distribution plot saved as  eda_distributions.png")

# ─────────────────────────────────────────
# 1.6  Correlation Heatmap
# ─────────────────────────────────────────
plt.figure(figsize=(11, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', center=0,
    linewidths=0.5, cbar_kws={'shrink': 0.8}
)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("  📊  Correlation heatmap saved as  eda_correlation.png")

# ─────────────────────────────────────────
# 1.7  Price vs Median Income scatter
# ─────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.scatter(df['MedInc'], df['Price'], alpha=0.25, s=8, color='#4C72B0')
plt.xlabel('Median Income (tens of thousands $)', fontsize=11)
plt.ylabel('Median House Value ($100k)', fontsize=11)
plt.title('Median Income vs House Price', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_income_vs_price.png', dpi=150, bbox_inches='tight')
plt.show()
print("  📊  Income vs Price scatter saved as  eda_income_vs_price.png")

print("\n\n✅  STEP 1 COMPLETE — EDA finished.\n")