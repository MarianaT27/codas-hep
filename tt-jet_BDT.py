import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from jet_util import load_processed_data, load_processed_data_extended
from jet_plotting_util import plot_confusion_matrix, plot_roc_curve, plot_feature_distributions, plot_loss_curve

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
# load_processed_data() (jet_util.py) reads the pre-computed per-jet cluster
# features (cluster_features.csv), labels (labels.npy, 0=QCD jet, 1=ttbar
# jet) and event ids for the train/val/test splits under
# qcd-tt-jet-tagging-co-da-s-he/. X_* are pandas DataFrames of features,
# y_* are the 0/1 labels, and *_ids let you trace a row back to its
# original event.
X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids = load_processed_data()

X_train.shape  # (n_train_jets, n_features) -- sanity check on row/column count

X_train.head()  # peek at the first few rows / column names

# ---------------------------------------------------------------------------
# 1a. Drop uninformative features
# ---------------------------------------------------------------------------
# The clustering step (jet_util.anti_kt_clustering) always stops once each
# cluster has exactly 2 particles, so max/mean/std_cluster_size and
# cluster_size_ratio are constant across every jet -- they carry zero
# information (confirmed by feature_importances_ == 0 for all four below).
CONSTANT_FEATURES = ['max_cluster_size', 'mean_cluster_size',
                      'std_cluster_size', 'cluster_size_ratio', 'max_cluster_pt']
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

# delta_r_leading on its own turned out to be 60-97% correlated with
# mean_cluster_phi/max_cluster_phi/std_cluster_phi already in the model --
# it looked separating in isolation but added no accuracy/F1/AUC because it
# was redundant, not because it lacked signal. Decorrelating it (regress
# out what mean_cluster_phi/max_cluster_phi already explain, keep only the
# residual) recovers the non-redundant part: R^2=0.43, i.e. 57% of
# delta_r_leading's variance is NOT explained by those phi features, and
# that residual still separates QCD/ttbar cleanly. Fit on train only and
# applied to val/test to avoid leaking val/test info into the regression.
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
# One histogram per feature (small multiples), QCD and ttbar overlaid in a
# fixed blue/red color pair. Using train+val together (both have labels;
# test does not) for more statistics per histogram. This is a quick way to
# see which features actually separate the two classes -- a feature whose
# QCD/ttbar histograms overlap almost completely isn't pulling weight, see
# feature_importances_ below for the model's own ranking.
train_val_df = pd.concat([X_train, X_val], ignore_index=True)
train_val_df['label'] = np.concatenate([y_train, y_val])
plot_feature_distributions(train_val_df, 'QCD vs ttbar Feature Distributions',output_dir='processed_data_BDT')

# ---------------------------------------------------------------------------
# 2. Initialize the model
# ---------------------------------------------------------------------------
# XGBClassifier is a gradient-boosted decision tree ensemble: it builds
# trees one at a time, each one correcting the errors of the trees before
# it. See "Tunable parameters" below for what each argument controls.

# scale_pos_weight counters class imbalance (train is 2308 QCD vs 1212
# ttbar, ~1.9:1) by upweighting positive-class (ttbar) errors during
# training -- the standard ratio is negative_count / positive_count.
# Without it, the model has less incentive to get the minority class right,
# which is exactly what F1 on ttbar is sensitive to.
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=45,              # number of boosting rounds (trees) to build -- val logloss bottoms out ~round 30-40, then flattens/rises; 60 gives headroom without the full 100 rounds of pure overfitting
    max_depth=5,                  # max depth of each tree (controls model complexity) -- lowered from 5, small dataset (3520 rows) overfits fast at depth 5
    learning_rate=0.1,            # shrinks each tree's contribution (a.k.a. eta)
    objective='binary:logistic',  # binary classification -> outputs a probability
    scale_pos_weight=scale_pos_weight,  # ~1.9, upweights ttbar errors to counter class imbalance
    subsample=0.9,                # each tree trains on a random 90% of rows -- reduces overfitting; 0.8 was tried first and slightly hurt F1 (0.8533->0.8486), likely too aggressive a cut on only 3520 training rows
    colsample_bytree=0.9,         # each tree only sees a random 90% of columns -- std_cluster_phi was eating 68% of feature importance on its own; this forces some trees to split on other features instead
    random_state=42               # fixes randomness so runs are reproducible
)

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
# eval_set with BOTH train and val lets us plot train-vs-val logloss below --
# that gap (or lack of one) is what actually tells you whether n_estimators
# and learning_rate are well matched: val loss turning back up while train
# loss keeps falling is the overfitting signal to watch for.
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_val, y_val)],
          verbose=True)

# Plot train vs. val logloss per boosting round.
plot_loss_curve(model.evals_result(), metric='logloss', output_dir='processed_data_BDT')

# Which features the trees actually relied on -- a quick, data-driven way
# to decide what to keep/drop if you want to trim X_train's columns.
importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("\nFeature importances (highest first):")
print(importances.sort_values(ascending=False))

# ---------------------------------------------------------------------------
# 4. Predict on the validation set
# ---------------------------------------------------------------------------
# predict_proba returns a probability for each class (columns: [P(QCD),
# P(ttbar)]). Since the two columns always sum to 1, we only need one of
# them -- column 1 is P(this jet is ttbar).
y_pred = model.predict_proba(X_val)[:, 1]

# Turn probabilities into hard 0/1 labels using a 0.5 decision threshold,
# needed for accuracy_score and the confusion matrix.
discrete_pred = np.where(y_pred > 0.5, 1, 0)

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
accuracy = accuracy_score(y_val, discrete_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# F1 = harmonic mean of precision and recall -- a better summary than
# accuracy here since the classes are imbalanced (2308 QCD vs 1212 ttbar
# in train), where accuracy alone can look good just by favoring the
# majority class.
f1 = f1_score(y_val, discrete_pred)
print(f"Test F1 Score: {f1:.4f}")

# Confusion matrix: rows = true label, columns = predicted label.
plot_confusion_matrix(y_val, discrete_pred, title='BDT Confusion Matrix')

# ROC curve: true positive rate vs. false positive rate as the decision
# threshold is swept, plus the AUC (area under the curve) summary metric.
# Uses the raw probabilities (y_pred), not the thresholded discrete_pred,
# since it needs to evaluate every possible threshold.
plot_roc_curve(y_val, y_pred, title='BDT ROC Curve')


test_predictions = model.predict_proba(X_test)[:, 1]
solution = pd.DataFrame({'id':test_ids, 'label':test_predictions})
solution.to_csv('solutions_BDT.csv', index=False)