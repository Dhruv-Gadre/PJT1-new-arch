from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from custom_espectre.data_io import load_dataset
from custom_espectre.features import extract_features


DATA_DIR = Path("data/raw")

windows, phases, labels, _ = load_dataset(DATA_DIR)

X = np.asarray([
    extract_features(amp, pha)
    for amp, pha in zip(windows, phases)
])

y = np.asarray(labels)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

all_true = []
all_pred = []
fold_accuracies = []

print("\n========================================")
print("5-FOLD CLASS-LEVEL EVALUATION")
print("========================================")

print(f"Total samples : {len(y)}")
print(f"Idle samples  : {np.sum(y == 0)}")
print(f"Motion samples: {np.sum(y == 1)}")

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = np.mean(pred == y_test)

    all_true.extend(y_test)
    all_pred.extend(pred)
    fold_accuracies.append(accuracy)

    print(f"\nFold {fold}")
    print(f"  Accuracy: {accuracy:.3f}")

all_true = np.asarray(all_true)
all_pred = np.asarray(all_pred)

precision, recall, f1, support = precision_recall_fscore_support(
    all_true,
    all_pred,
    labels=[0, 1],
    zero_division=0,
)

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1],
)

print("\n========================================")
print("OVERALL 5-FOLD RESULTS")
print("========================================")

print(f"Mean accuracy: {np.mean(fold_accuracies):.3f}")
print(f"Std accuracy : {np.std(fold_accuracies):.3f}")

print("\nClass-level metrics:")
print("----------------------------------------")

print("Idle")
print(f"  Precision : {precision[0]:.3f}")
print(f"  Recall    : {recall[0]:.3f}")
print(f"  F1-score  : {f1[0]:.3f}")
print(f"  Support   : {support[0]}")

print("\nMotion")
print(f"  Precision : {precision[1]:.3f}")
print(f"  Recall    : {recall[1]:.3f}")
print(f"  F1-score  : {f1[1]:.3f}")
print(f"  Support   : {support[1]}")

print("\nConfusion matrix:")
print("                 Predicted")
print("              Idle   Motion")
print(f"Actual Idle   {cm[0,0]:5d}   {cm[0,1]:6d}")
print(f"Actual Motion {cm[1,0]:5d}   {cm[1,1]:6d}")

print("\n========================================")
print("DONE")
print("========================================")