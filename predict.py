"""
predict.py

    python predict.py some_image.jpg
    -> prints a single float in [0, 1]:  0 = real photo, 1 = photo of a screen

Loads weights.json (produced by train.py on your own real/screen photos) if
present next to this file. If it's missing, falls back to a hand-tuned
linear rule built from domain knowledge (no training data needed at all,
per the assignment's "training a model is NOT required") -- so this script
always works out of the box.
"""

import json
import os
import sys
import time

import numpy as np

from features import extract_features, FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "weights.json")

# Hand-tuned fallback (used only if weights.json doesn't exist yet).
# Signs come directly from the feature docstrings in features.py:
# all six features are designed to increase for screens/recaptures.
FALLBACK_COEF = {
    "moire_peak_ratio": 0.9,
    "hf_energy_ratio": 4.0,
    "specular_fraction": 6.0,
    "border_line_score": 0.8,
    "banding_score": 5.0,
    "laplacian_var_norm": 0.05,
}
FALLBACK_MEAN = {
    "moire_peak_ratio": 3.0,
    "hf_energy_ratio": 0.15,
    "specular_fraction": 0.02,
    "border_line_score": 0.3,
    "banding_score": 0.15,
    "laplacian_var_norm": 1.0,
}
FALLBACK_INTERCEPT = -3.2


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_weights():
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH) as f:
            w = json.load(f)
        return (np.array(w["mean"]), np.array(w["std"]),
                np.array(w["coef"]), w["intercept"], "trained")
    else:
        mean = np.array([FALLBACK_MEAN[n] for n in FEATURE_NAMES])
        std = np.ones(len(FEATURE_NAMES))
        coef = np.array([FALLBACK_COEF[n] for n in FEATURE_NAMES])
        return mean, std, coef, FALLBACK_INTERCEPT, "fallback"


def predict(image_path, weights=None):
    if weights is None:
        weights = load_weights()
    mean, std, coef, intercept, mode = weights

    feats = extract_features(image_path)
    z = (feats - mean) / std
    score = _sigmoid(np.dot(z, coef) + intercept)
    return float(score), mode


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [--benchmark]")
        sys.exit(1)

    image_path = sys.argv[1]
    benchmark = "--benchmark" in sys.argv

    weights = load_weights()

    if benchmark:
        # warm-up (import/JIT-ish overhead should not count against latency)
        predict(image_path, weights)
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            score, mode = predict(image_path, weights)
        t1 = time.perf_counter()
        print(f"score={score:.4f} mode={mode} avg_latency_ms={(t1 - t0) / N * 1000:.2f}")
    else:
        score, mode = predict(image_path, weights)
        print(f"{score:.4f}")


if __name__ == "__main__":
    main()
