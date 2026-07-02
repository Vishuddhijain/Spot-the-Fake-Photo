"""
train.py

Fits a tiny logistic regression (6 features -> 1 weight per feature + bias)
on your own real/ and screen/ photo folders. Training is NOT required by
the assignment (predict.py has a sensible hand-tuned fallback), but a
2-minute fit on your own ~100 photos meaningfully improves accuracy and
still keeps the model tiny (7 floats total) and fast.

Usage:
    python train.py --real_dir real/ --screen_dir screen/ --out weights.json

Outputs weights.json: {"mean": [...], "std": [...], "coef": [...], "intercept": ...}
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report

from features import extract_features, FEATURE_NAMES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in IMG_EXTS]


def build_dataset(real_dir, screen_dir):
    X, y, paths = [], [], []
    for p in list_images(real_dir):
        try:
            X.append(extract_features(p)); y.append(0); paths.append(p)
        except Exception as e:
            print(f"  skip {p}: {e}", file=sys.stderr)
    for p in list_images(screen_dir):
        try:
            X.append(extract_features(p)); y.append(1); paths.append(p)
        except Exception as e:
            print(f"  skip {p}: {e}", file=sys.stderr)
    return np.array(X, dtype=np.float32), np.array(y), paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="real")
    ap.add_argument("--screen_dir", default="screen")
    ap.add_argument("--out", default="weights.json")
    args = ap.parse_args()

    print(f"Loading images from '{args.real_dir}' (label=REAL) and '{args.screen_dir}' (label=SCREEN)...")
    X, y, paths = build_dataset(args.real_dir, args.screen_dir)
    print(f"Loaded {len(y)} images: {int((y==0).sum())} real, {int((y==1).sum())} screen.")

    if len(y) < 20 or len(set(y.tolist())) < 2:
        print("Not enough labeled data in both folders (need real/ and screen/ each with photos). "
              "predict.py will fall back to the hand-tuned rule.")
        return

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xn = (X - mean) / std

    # 5-fold CV so the reported accuracy is honest (not just train-set fit)
    n_splits = min(5, np.bincount(y).min())
    n_splits = max(n_splits, 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    clf_cv = LogisticRegression(max_iter=2000, class_weight="balanced")
    y_pred_cv = cross_val_predict(clf_cv, Xn, y, cv=skf)
    cv_acc = accuracy_score(y, y_pred_cv)

    print(f"\n{n_splits}-fold cross-validated accuracy: {cv_acc*100:.1f}%")
    print(classification_report(y, y_pred_cv, target_names=["real", "screen"]))

    # final model fit on ALL data, to ship
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xn, y)

    for name, w in zip(FEATURE_NAMES, clf.coef_[0]):
        print(f"  {name:22s} weight={w:+.3f}")

    weights = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "feature_names": FEATURE_NAMES,
        "cv_accuracy": cv_acc,
        "n_train": len(y),
    }
    with open(args.out, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\nSaved trained weights to {args.out}")
    print("Report this cv_accuracy number in your note (it's the honest, held-out-style estimate).")


if __name__ == "__main__":
    main()
