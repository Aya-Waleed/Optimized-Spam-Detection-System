"""
Shared spam-detection system logic used by deploy/app.py.

Loads the final deployment pipeline (models/spam_detection_pipeline.pkl),
its configuration (models/deployment_config.json), and — for prediction
explanations — the interpretable (non-calibrated) tuned SVM together
with the TF-IDF vectorizer from outputs/.
"""

import json
import os

import joblib
import numpy as np

from text_preprocessing import preprocess_text

_HERE = os.path.dirname(__file__)

MODEL_PATH = os.path.join(_HERE, "..", "models", "spam_detection_pipeline.pkl")
CONFIG_PATH = os.path.join(_HERE, "..", "models", "deployment_config.json")

TFIDF_PATH = os.path.join(_HERE, "..", "outputs", "tfidf_vectorizer.pkl")
SVM_PATH = os.path.join(_HERE, "..", "outputs", "best_svm_model.pkl")


def load_pipeline():
    """Load the deployable pipeline and its configuration."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        return None, None

    pipeline = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return pipeline, config


def load_explainer():
    """Load the interpretable TF-IDF + linear SVM pair used to explain predictions.

    Returns (None, None) if the artifacts are missing, so the caller can
    degrade gracefully (predictions still work; explanations are skipped).
    """
    if not os.path.exists(TFIDF_PATH) or not os.path.exists(SVM_PATH):
        return None, None

    tfidf = joblib.load(TFIDF_PATH)
    svm_model = joblib.load(SVM_PATH)
    return tfidf, svm_model


def classify_one(pipeline, config, message: str, threshold: float = None) -> dict:
    """Classify a single message. `threshold` overrides the saved default when given."""
    message = str(message).strip()
    if not message:
        raise ValueError("Message is empty.")

    used_threshold = threshold if threshold is not None else config["threshold"]
    label_mapping = config["label_mapping"]

    proba = float(pipeline.predict_proba([message])[0, 1])
    predicted_class = 1 if proba >= used_threshold else 0

    return {
        "message": message,
        "label": label_mapping[str(predicted_class)],
        "spam_probability": round(proba, 4),
        "threshold_used": used_threshold,
    }


def classify_many(pipeline, config, messages, threshold: float = None):
    """Classify a list/Series of messages. Skips empty/invalid entries gracefully."""
    used_threshold = threshold if threshold is not None else config["threshold"]
    label_mapping = config["label_mapping"]

    cleaned = [str(m).strip() if str(m).strip() else "" for m in messages]
    non_empty_idx = [i for i, m in enumerate(cleaned) if m]

    probabilities = np.zeros(len(cleaned))
    if non_empty_idx:
        proba_values = pipeline.predict_proba([cleaned[i] for i in non_empty_idx])[:, 1]
        for i, p in zip(non_empty_idx, proba_values):
            probabilities[i] = p

    results = []
    for i, message in enumerate(cleaned):
        if i not in non_empty_idx:
            results.append({"message": messages[i], "label": None, "spam_probability": None})
            continue

        predicted_class = 1 if probabilities[i] >= used_threshold else 0
        results.append({
            "message": messages[i],
            "label": label_mapping[str(predicted_class)],
            "spam_probability": round(float(probabilities[i]), 4),
        })

    return results


def explain_prediction(tfidf, svm_model, message: str, top_n: int = 6):
    """Return the top contributing words for a message, using the linear SVM's coefficients.

    Positive contribution pushes the message toward "spam", negative
    toward "ham". Only words that are both present in the message and
    part of the TF-IDF vocabulary can be explained.
    """
    processed = preprocess_text(message)
    vector = tfidf.transform([processed])
    coefficients = svm_model.coef_[0]

    feature_names = tfidf.get_feature_names_out()
    nonzero_indices = vector.nonzero()[1]

    contributions = []
    for idx in nonzero_indices:
        contribution = float(vector[0, idx] * coefficients[idx])
        contributions.append((feature_names[idx], contribution))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]
