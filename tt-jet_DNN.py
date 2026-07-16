import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from jet_util import load_processed_data, load_processed_data_extended
from jet_plotting_util import (plot_training_history, plot_confusion_matrix,
                                plot_roc_curve, plot_feature_distributions)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
# Same feature set as tt-jet_BDT.py, so the two models are directly
# comparable -- see that script for the reasoning behind each step below.
X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids = load_processed_data()

X_train.shape  # (n_train_jets, n_features) -- sanity check on row/column count

X_train.head()  # peek at the first few rows / column names

# ---------------------------------------------------------------------------
# 1a. Drop uninformative features
# ---------------------------------------------------------------------------
# The clustering step (jet_util.anti_kt_clustering) always stops once each
# cluster has exactly 2 particles, so max/mean/std_cluster_size and
# cluster_size_ratio are constant across every jet -- they carry zero
# information.
CONSTANT_FEATURES = ['max_cluster_size', 'mean_cluster_size',
                      'std_cluster_size', 'cluster_size_ratio']
X_train = X_train.drop(columns=CONSTANT_FEATURES)
X_val = X_val.drop(columns=CONSTANT_FEATURES)
X_test = X_test.drop(columns=CONSTANT_FEATURES)


# ---------------------------------------------------------------------------
# 1a-2. Engineered shape features, computed from the raw jet images
# ---------------------------------------------------------------------------
# load_processed_data_extended() reruns anti_kt_clustering over the jet
# images and recomputes extract_cluster_features(), which now also returns:
#   std_cluster_phi     -- spread of cluster positions in phi (jet shape)
#   eta_phi_asymmetry   -- (std_eta-std_phi)/(std_eta+std_phi), bounded [-1,1]
#   delta_r_leading     -- angular separation between the two leading clusters
# std_cluster_eta was dropped: its QCD/ttbar histograms overlapped almost
# completely (no separating power).
X_train_ext, _, _, X_val_ext, _, _, X_test_ext, _ = load_processed_data_extended()

# delta_r_leading on its own is 60-97% correlated with mean_cluster_phi /
# max_cluster_phi / std_cluster_phi already in the feature set. Decorrelating
# it (regress out what those phi features already explain, keep only the
# residual) recovers the non-redundant part: R^2=0.43, i.e. 57% of
# delta_r_leading's variance is NOT explained by the phi features. Fit on
# train only and applied to val/test to avoid leaking val/test info in.
phi_predictors = ['mean_cluster_phi', 'max_cluster_phi']
decorrelator = LinearRegression()
decorrelator.fit(X_train[phi_predictors], X_train_ext['delta_r_leading'])

def residualize(X_base, X_ext, column):
    predicted = decorrelator.predict(X_base[phi_predictors])
    return X_ext[column] - predicted

X_train['delta_r_leading_resid'] = residualize(X_train, X_train_ext, 'delta_r_leading')
X_val['delta_r_leading_resid'] = residualize(X_val, X_val_ext, 'delta_r_leading')
X_test['delta_r_leading_resid'] = residualize(X_test, X_test_ext, 'delta_r_leading')

NEW_FEATURES = ['std_cluster_phi', 'eta_phi_asymmetry']
X_train = pd.concat([X_train, X_train_ext[NEW_FEATURES]], axis=1)
X_val = pd.concat([X_val, X_val_ext[NEW_FEATURES]], axis=1)
X_test = pd.concat([X_test, X_test_ext[NEW_FEATURES]], axis=1)

# ---------------------------------------------------------------------------
# 1b. Explore the features: QCD vs ttbar distributions
# ---------------------------------------------------------------------------
train_val_df = pd.concat([X_train, X_val], ignore_index=True)
train_val_df['label'] = np.concatenate([y_train, y_val])
plot_feature_distributions(train_val_df, 'QCD vs ttbar Feature Distributions (DNN)')

# ---------------------------------------------------------------------------
# 1c. Scale features
# ---------------------------------------------------------------------------
# Unlike trees (BDT), a neural net's gradient descent is sensitive to
# feature scale -- a feature ranging in the thousands would dominate the
# loss surface over one ranging in [0,1] regardless of actual importance.
# StandardScaler rescales every column to mean 0 / std 1.
# IMPORTANT: fit the scaler on train only, then just .transform() val/test
# with those same train statistics -- fitting separately on val/test (as
# the original notebook did with X_test_scaled = scaler.fit_transform(X_test))
# leaks each split's own distribution into itself and silently makes
# train/val/test incomparable.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 2. Build the DNN model
# ---------------------------------------------------------------------------
# A small fully-connected network: each Dense layer's neuron count is a
# tunable choice (conventionally powers of 2), Dropout randomly zeroes out
# a fraction of activations during training as a regularizer (same
# overfitting concern as max_depth/subsample in the BDT), and the final
# single sigmoid neuron outputs a ttbar probability, mirroring
# predict_proba(...)[:, 1] in the BDT script.
def build_dnn_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),

        # single output neuron + sigmoid -- P(ttbar), same role as
        # XGBClassifier's predict_proba(...)[:, 1] in the BDT script.
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

model = build_dnn_model(X_train_scaled.shape[1])
model.summary()

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
# class_weight counters the same class imbalance (2308 QCD vs 1212 ttbar in
# train) that scale_pos_weight handles in the BDT script -- it upweights
# ttbar's contribution to the loss so the network doesn't just lean on the
# majority class.
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
class_weight = {0: 1.0, 1: neg / pos}

# EarlyStopping(patience=5) stops training once val_loss hasn't improved
# for 5 epochs and restores the best-epoch weights -- the DNN equivalent of
# picking n_estimators from the BDT's loss curve, done automatically.
# validation_data=(X_val_scaled, y_val) uses our actual held-out val split
# (rather than the notebook's validation_split=0.2, which would carve the
# split out of X_train instead and never touch X_val until final eval).
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,          # max passes over the full training set
    batch_size=32,       # samples per gradient update
    validation_data=(X_val_scaled, y_val),
    class_weight=class_weight,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# Plot training/validation loss and accuracy per epoch -- same idea as
# plot_loss_curve() for the BDT, watch for val diverging from train.
plot_training_history(history, metrics=['loss', 'accuracy'])

# ---------------------------------------------------------------------------
# 4. Predict on the validation set
# ---------------------------------------------------------------------------
y_pred = model.predict(X_val_scaled).flatten()  # P(ttbar) per jet

# Turn probabilities into hard 0/1 labels using a 0.5 decision threshold.
discrete_pred = (y_pred > 0.5).astype(int)

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
accuracy = accuracy_score(y_val, discrete_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# F1 = harmonic mean of precision and recall -- a better summary than
# accuracy here since the classes are imbalanced, same reasoning as the
# BDT script.
f1 = f1_score(y_val, discrete_pred)
print(f"Test F1 Score: {f1:.4f}")

# Confusion matrix: rows = true label, columns = predicted label.
plot_confusion_matrix(y_val, discrete_pred, title='DNN Confusion Matrix')

# ROC curve: true positive rate vs. false positive rate as the decision
# threshold is swept, plus the AUC (area under the curve) summary metric.
plot_roc_curve(y_val, y_pred, title='DNN ROC Curve')

# ---------------------------------------------------------------------------
# 6. Make predictions on the test set (uncomment to generate a submission)
# ---------------------------------------------------------------------------
test_predictions = model.predict(X_test_scaled)
solution = pd.DataFrame({'id': test_ids, 'label': test_predictions[:, 0]})
solution.to_csv('submission_DNN.csv', index=False)
