from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _positive_class_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError("Expected binary predict_proba output with two columns.")
        return np.asarray(probs[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    raise ValueError("Model must expose predict_proba or decision_function for calibration.")


class ProbabilityCalibrator:
    def __init__(self, method: str = "platt"):
        method_norm = str(method).strip().lower()
        if method_norm not in {"platt", "isotonic"}:
            raise ValueError("Calibration method must be 'platt' or 'isotonic'.")
        self.method = method_norm
        self.model_: LogisticRegression | IsotonicRegression | None = None

    def fit(self, raw_scores, y_true) -> "ProbabilityCalibrator":
        scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        y = np.asarray(y_true, dtype=int)
        if len(scores) != len(y):
            raise ValueError("Calibration scores and labels must be the same length.")
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration requires both positive and negative labels.")

        if self.method == "platt":
            calibrator = LogisticRegression(max_iter=1000, random_state=42)
            calibrator.fit(scores, y)
            self.model_ = calibrator
        else:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(scores.ravel(), y)
            self.model_ = calibrator
        return self

    def predict_proba(self, raw_scores) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("ProbabilityCalibrator must be fit before prediction.")
        scores = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        if self.method == "platt":
            pos = self.model_.predict_proba(scores)[:, 1]
        else:
            pos = self.model_.predict(scores.ravel())
        pos = np.clip(np.asarray(pos, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - pos, pos])


@dataclass
class CalibratedModel:
    base_model: Any
    calibrator: ProbabilityCalibrator | None = None
    calibration_metadata: dict[str, Any] | None = None

    def predict_proba_raw(self, X) -> np.ndarray:
        raw_scores = _positive_class_scores(self.base_model, X)
        raw_scores = np.clip(raw_scores, 0.0, 1.0)
        return np.column_stack([1.0 - raw_scores, raw_scores])

    def predict_proba(self, X) -> np.ndarray:
        raw_scores = _positive_class_scores(self.base_model, X)
        if self.calibrator is None:
            raw_scores = np.clip(raw_scores, 0.0, 1.0)
            return np.column_stack([1.0 - raw_scores, raw_scores])
        return self.calibrator.predict_proba(raw_scores)

