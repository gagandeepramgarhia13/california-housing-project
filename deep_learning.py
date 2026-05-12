# ============================================================
# STEP 4: Deep Learning Model — TensorFlow / Keras
# California Housing Price Prediction Project
# ============================================================
# Run AFTER step2_preprocessing.py
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import joblib, os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress TF INFO logs

import tensorflow as tf
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics            import mean_absolute_error, mean_squared_error, r2_score

print("=" * 55)
print("   STEP 4: DEEP LEARNING MODEL")
print(f"   TensorFlow version: {tf.__version__}")
print("=" * 55)

# ─────────────────────────────────────────
# 4.1  Load preprocessed data
# ─────────────────────────────────────────
X_train = np.load('models/X_train.npy')
X_test  = np.load('models/X_test.npy')
y_train = np.load('models/y_train.npy')
y_test  = np.load('models/y_test.npy')

print(f"\n  X_train shape: {X_train.shape}")
print(f"  X_test  shape: {X_test.shape}")

n_features = X_train.shape[1]

# ─────────────────────────────────────────
# 4.2  Build the Neural Network
# ─────────────────────────────────────────
print("\n  Building Neural Network architecture...")

tf.random.set_seed(42)
np.random.seed(42)

model = Sequential([
    # Input + first hidden layer
    Dense(256, activation='relu', input_shape=(n_features,),
          kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.3),

    # Second hidden layer
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.2),

    # Third hidden layer
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.2),

    # Fourth hidden layer
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.1),

    # Output (regression — single neuron, no activation)
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n  Model Summary:")
model.summary()

# ─────────────────────────────────────────
# 4.3  Callbacks
# ─────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=7, min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(
        'models/deep_learning_best.keras',
        monitor='val_loss', save_best_only=True, verbose=0
    )
]

# ─────────────────────────────────────────
# 4.4  Training
# ─────────────────────────────────────────
print("\n\n  Training started (up to 100 epochs, early stopping on)...")
print("  ─" * 28)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

print("\n  Training complete.")
print(f"  Stopped at epoch : {len(history.history['loss'])}")

# ─────────────────────────────────────────
# 4.5  Evaluation
# ─────────────────────────────────────────
print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  DEEP LEARNING — TEST SET EVALUATION")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

pred_dl     = model.predict(X_test, verbose=0).flatten()
mae_dl      = mean_absolute_error(y_test, pred_dl)
rmse_dl     = np.sqrt(mean_squared_error(y_test, pred_dl))
r2_dl       = r2_score(y_test, pred_dl)

print(f"  MAE       : {mae_dl:.4f}  (× $100k = ${mae_dl*100:.0f}k avg error)")
print(f"  RMSE      : {rmse_dl:.4f}")
print(f"  R² Score  : {r2_dl:.4f}")

# ─────────────────────────────────────────
# 4.6  Training History Plot
# ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Deep Learning — Training History', fontsize=14, fontweight='bold')

epochs_range = range(1, len(history.history['loss']) + 1)

ax1.plot(epochs_range, history.history['loss'],     label='Train Loss', color='#4C72B0', linewidth=1.5)
ax1.plot(epochs_range, history.history['val_loss'], label='Val Loss',   color='#DD8452', linewidth=1.5)
ax1.set_title('Loss (MSE)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history.history['mae'],     label='Train MAE', color='#4C72B0', linewidth=1.5)
ax2.plot(epochs_range, history.history['val_mae'], label='Val MAE',   color='#DD8452', linewidth=1.5)
ax2.set_title('MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('deep_learning_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  📊 Training history saved as deep_learning_history.png")

# ─────────────────────────────────────────
# 4.7  Actual vs Predicted scatter
# ─────────────────────────────────────────
plt.figure(figsize=(7, 6))
plt.scatter(y_test, pred_dl, alpha=0.25, s=8, color='#7F77DD')
mn = min(y_test.min(), pred_dl.min())
mx = max(y_test.max(), pred_dl.max())
plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Price ($100k)', fontsize=11)
plt.ylabel('Predicted Price ($100k)', fontsize=11)
plt.title(f'Deep Learning — Actual vs Predicted  (R²={r2_dl:.3f})', fontsize=12, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('deep_learning_actual_vs_pred.png', dpi=150, bbox_inches='tight')
plt.show()
print("  📊 Actual vs Predicted saved as deep_learning_actual_vs_pred.png")

# ─────────────────────────────────────────
# 4.8  Save model
# ─────────────────────────────────────────
model.save('models/deep_learning_model.keras')
print("\n  ✅ Full model saved: models/deep_learning_model.keras")
print("  ✅ Best weights  at: models/deep_learning_best.keras")

print("\n\n✅  STEP 4 COMPLETE — Deep Learning model trained & saved.\n")