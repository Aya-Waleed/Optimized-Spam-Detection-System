"""
Sanity check for the deployed spam detection pipeline.

This is not meant to replace proper evaluation on the held-out test
split (already covered in the notebooks). It is a quick smoke test that
loads the saved pipeline and checks it against a handful of hand-written
messages that are NOT part of the SMS Spam Collection dataset, to catch
obvious deployment issues (missing files, broken pipeline, wrong label
mapping) and get a rough sense of generalization to new text.

Usage
-----
    python tests/test_pipeline_sanity.py
"""

import os
import sys

# Allow importing predict_spam-style loading logic without duplicating it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))

from predict_cli import load_system, classify  # noqa: E402

# Hand-written examples, not taken from data/spam.csv
FRESH_EXAMPLES = [
    ("WINNER!! You have been selected to receive a FREE iPhone. Text CLAIM to 80099 now!", "spam"),
    ("Reminder: your dentist appointment is tomorrow at 10am.", "ham"),
    ("Your bank account has been locked. Verify your details immediately at secure-bank-login.com", "spam"),
    ("Can you pick up milk on your way home?", "ham"),
    ("Congrats! You've won a $1000 Walmart gift card. Click the link to claim your prize now!", "spam"),
    ("Mom, I'll be home late tonight, don't wait up for dinner.", "ham"),
    ("URGENT: Your subscription payment failed. Update your card details to avoid service interruption.", "spam"),
    ("Let's grab coffee this weekend, it's been a while!", "ham"),
]


def main():
    pipeline, config = load_system()

    correct = 0
    print("Sanity check on fresh, unseen messages")
    print("=" * 70)

    for message, expected in FRESH_EXAMPLES:
        result = classify(pipeline, config, message)
        is_correct = result["label"] == expected
        correct += int(is_correct)

        status = "OK  " if is_correct else "MISS"
        print(f"[{status}] expected={expected:<5} predicted={result['label']:<5} "
              f"(p={result['spam_probability']:.4f})  {message[:60]}")

    total = len(FRESH_EXAMPLES)
    print("=" * 70)
    print(f"Result: {correct}/{total} correct on fresh examples")

    if correct < total:
        print("Note: a few misses on hand-written examples are expected — "
              "this is a small, informal check, not a benchmark.")


if __name__ == "__main__":
    main()
