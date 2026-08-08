"""
Spam Detection - Command Line Interface

Loads the final deployment pipeline produced in
notebooks/Model_Optimization_Deployment.ipynb and classifies SMS
messages as "spam" or "ham".

Usage
-----
Interactive mode:
    python deploy/predict_cli.py

Single message mode:
    python deploy/predict_cli.py "Congratulations! You won a free prize!"

This script must be run from the project root (the folder containing
`models/`, `deploy/`, `notebooks/`, etc.), or with paths adjusted
accordingly.
"""

import json
import os
import sys

import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "spam_detection_pipeline.pkl")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "deployment_config.json")


def load_system():
    """Load the trained pipeline and its deployment configuration."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            "Deployment artifacts not found. Run the "
            "'Model_Optimization_Deployment.ipynb' notebook first to "
            "generate 'models/spam_detection_pipeline.pkl' and "
            "'models/deployment_config.json'."
        )

    pipeline = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return pipeline, config


def classify(pipeline, config, message: str) -> dict:
    """Classify a single message and return label + spam probability."""
    if not isinstance(message, str):
        raise TypeError(f"Expected a string message, got {type(message).__name__}")

    message = message.strip()
    if not message:
        raise ValueError("Message is empty. Please provide some text to classify.")

    threshold = config["threshold"]
    label_mapping = config["label_mapping"]

    proba = pipeline.predict_proba([message])[0, 1]
    predicted_class = 1 if proba >= threshold else 0

    return {
        "message": message,
        "label": label_mapping[str(predicted_class)],
        "spam_probability": round(float(proba), 4),
    }


def print_result(result: dict) -> None:
    tag = "🚫 SPAM" if result["label"] == "spam" else "✅ HAM"
    display_message = result["message"]
    if len(display_message) > 200:
        display_message = display_message[:200] + "..."

    print(f"\n{tag}  (spam probability: {result['spam_probability']:.2%})")
    print(f"Message: {display_message}\n")


def run_interactive(pipeline, config) -> None:
    print("=" * 60)
    print("Spam Detection System - Interactive Mode")
    print("Type a message and press Enter to classify it.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            message = input("\nEnter a message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if message.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not message:
            print("Please enter a non-empty message.")
            continue

        try:
            result = classify(pipeline, config, message)
        except (TypeError, ValueError) as exc:
            print(f"Could not classify that message: {exc}")
            continue

        print_result(result)


def main():
    try:
        pipeline, config = load_system()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if len(sys.argv) > 1:
        # Single message passed as a command-line argument
        message = " ".join(sys.argv[1:])
        try:
            result = classify(pipeline, config, message)
        except (TypeError, ValueError) as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        print_result(result)
    else:
        run_interactive(pipeline, config)


if __name__ == "__main__":
    main()
