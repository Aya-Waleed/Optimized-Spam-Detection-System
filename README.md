# Optimized Spam Detection System

An end-to-end SMS spam detection system built with classic NLP and machine
learning: raw text messages go in, and a spam/ham prediction with a
confidence score comes out.

The project is organized as three connected stages, each owned by a
different notebook (or script), passing its output to the next stage
through the `outputs/` and `models/` folders.

```
Raw SMS text  ─▶  Data & NLP Pipeline  ─▶  ML & Model Evaluation  ─▶  Model Optimization & Deployment  ─▶  Usable system
```

## Team — Division of Work

| Stage | Notebook / Component | Responsible |
|---|---|---|
| 1. Data & NLP Pipeline | `notebooks/Spam_Detection.ipynb` | Aya Waleed Mohamed Mohamed Shehata |
| 2. ML & Model Evaluation | `notebooks/Model_Training_Evaluation.ipynb` | Heba Abdelsalam Elsayed |
| 3. Model Optimization & Deployment | `notebooks/Model_Optimization_Deployment.ipynb`, `deploy/` | SeifEidSalem Seliman Mohsen |

## 1. Data & NLP Pipeline — `notebooks/Spam_Detection.ipynb`

Turns the raw SMS Spam Collection dataset into ML-ready numerical features.

- Loads `data/spam.csv`, drops unused columns, renames `v1`/`v2` to
  `label`/`message`
- Removes duplicate records (403 duplicates found)
- Exploratory Data Analysis: class balance, message length, word count
- Text preprocessing: lowercasing, punctuation removal, tokenization,
  stopword removal, stemming
- TF-IDF feature extraction (unigrams + bigrams, 5,000 features)
- Saves the cleaned dataset and the TF-IDF feature matrix

**Outputs:** `outputs/cleaned_dataset.csv`, `outputs/tfidf_features.npz`,
`outputs/tfidf_vectorizer.pkl`, `outputs/labels.npy`,
`outputs/tfidf_feature_names.csv`, `outputs/figures/*.png`

## 2. ML & Model Evaluation — `notebooks/Model_Training_Evaluation.ipynb`

Trains and compares several classifiers on the TF-IDF features.

- Trains Logistic Regression, Multinomial Naive Bayes, and Linear SVM
- Evaluates each with accuracy, precision, recall, F1-score, and a
  confusion matrix
- Tunes the best model (Linear SVM) with `GridSearchCV` (5-fold CV)
- Saves the tuned model

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.9681 | 0.9900 | 0.7557 | 0.8571 |
| Multinomial Naive Bayes | 0.9797 | 1.0000 | 0.8397 | 0.9129 |
| Linear SVM | 0.9865 | 0.9835 | 0.9084 | 0.9444 |
| **Tuned Linear SVM** | **0.9865** | **0.9835** | **0.9084** | **0.9444** |

**Outputs:** `outputs/best_svm_model.pkl`

## 3. Model Optimization & Deployment — `notebooks/Model_Optimization_Deployment.ipynb`

Turns the best model into a deployable system.

- Calibrates the tuned SVM (`CalibratedClassifierCV`) so it outputs a
  spam **probability**, not just a hard label
- Analyzes the precision-recall trade-off and picks a decision threshold
  that maximizes F1-score (0.4189, instead of the default 0.5)
- **Re-validates the model choice**: applies the same calibration +
  threshold tuning to Logistic Regression and Naive Bayes and confirms
  the SVM still wins on F1-score and ROC AUC, even when every model gets
  the same fair treatment
- **Feature importance analysis**: extracts the SVM's linear
  coefficients to show which words push a message toward spam vs. ham
- **Cross-validation robustness check**: 5-fold CV on the full raw-text
  pipeline (TF-IDF re-fit per fold, no leakage) confirms the F1-score is
  stable across different data splits (mean F1 ≈ 0.939, std ≈ 0.008)
- Combines the **preprocessing step, TF-IDF vectorizer, and calibrated
  model** into a single `Pipeline` object (raw text in, prediction out) —
  the same text cleaning used in training is applied automatically
- Saves the final pipeline, its configuration, and the feature
  importance data to `models/`
- Provides a reusable `predict_spam()` function and a command-line
  interface

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Tuned Linear SVM (threshold 0.5) | 0.9865 | 0.9835 | 0.9084 | 0.9444 |
| Calibrated SVM (threshold 0.5) | 0.9884 | 0.9760 | 0.9313 | 0.9531 |
| **Calibrated SVM (threshold 0.4189)** | **0.9913** | **0.9766** | **0.9542** | **0.9653** |

After optimizing all three models with their own tuned threshold, the
SVM is still the best:

| Model (optimized) | F1-Score | ROC AUC |
|---|---|---|
| Logistic Regression | 0.9531 | 0.9941 |
| Multinomial Naive Bayes | 0.9490 | 0.9873 |
| **Calibrated Linear SVM** | **0.9653** | **0.9960** |

**Outputs:** `models/spam_detection_pipeline.pkl`,
`models/deployment_config.json`, `models/feature_importance.json`,
`outputs/figures/feature_importance.png`

## Deployment

Two ways to use the final system are included in `deploy/`:

- **`deploy/predict_cli.py`** — command-line interface
- **`deploy/app.py`** — Streamlit web app, with:
  - A styled UI with a gradient header, dark sidebar, and card-style layout
  - A **decision threshold slider** to trade off precision vs. recall live
  - **Word-level explanations** for each prediction, highlighting which words pushed the message toward spam or ham
  - A **history** of the last 10 messages classified in the session
  - **Batch classification from a CSV upload**, with a results table and a download button

Both interfaces share the same trained pipeline (`models/spam_detection_pipeline.pkl`),
so predictions are always consistent between them.

`deploy/text_preprocessing.py` and `deploy/spam_system.py` hold shared
logic (text cleaning, loading, classification, explanation) reused by
the notebook, the CLI, the app, and the tests — so training-time and
deployment-time preprocessing never drift out of sync.

## Project Structure

```
.
├── data/
│   └── spam.csv                      # raw SMS Spam Collection dataset
├── notebooks/
│   ├── Spam_Detection.ipynb                    # Stage 1: Data & NLP Pipeline
│   ├── Model_Training_Evaluation.ipynb         # Stage 2: ML & Model Evaluation
│   └── Model_Optimization_Deployment.ipynb     # Stage 3: Optimization & Deployment
├── outputs/
│   ├── cleaned_dataset.csv
│   ├── tfidf_features.npz
│   ├── tfidf_vectorizer.pkl
│   ├── labels.npy
│   ├── tfidf_feature_names.csv
│   ├── best_svm_model.pkl
│   └── figures/                      # saved plots from EDA and evaluation
├── models/
│   ├── spam_detection_pipeline.pkl   # final deployable pipeline
│   └── deployment_config.json        # decision threshold + label mapping
├── deploy/
│   ├── predict_cli.py                # command-line interface
│   ├── app.py                        # Streamlit web app
│   ├── spam_system.py                # shared loading/classification/explanation logic
│   └── text_preprocessing.py         # shared text cleaning (training & deployment)
├── tests/
│   └── test_pipeline_sanity.py       # smoke test on fresh, unseen messages
├── requirements.txt

```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The first notebook downloads the required NLTK data (`punkt`, `punkt_tab`,
`stopwords`) automatically on first run.

## Running the Project

Run the notebooks in order, since each stage depends on the previous
stage's output files:

1. `notebooks/Spam_Detection.ipynb`
2. `notebooks/Model_Training_Evaluation.ipynb`
3. `notebooks/Model_Optimization_Deployment.ipynb`

Once `models/spam_detection_pipeline.pkl` exists, the system can be used
directly without re-running any notebook:

```bash
# Command-line interface — interactive mode
python deploy/predict_cli.py

# Command-line interface — single message
python deploy/predict_cli.py "Congratulations! You won a free prize!"

# Streamlit web app
streamlit run deploy/app.py
```

**Important:** run this command from the project root (the folder that
contains `models/`, `deploy/`, etc.), not from inside `deploy/` — the
app looks for `models/spam_detection_pipeline.pkl` relative to the
project root. If the command runs but nothing opens in your browser,
copy the `Local URL` printed in the terminal (e.g. `http://localhost:8501`)
and open it manually.

## Sanity Check on Unseen Messages

`tests/test_pipeline_sanity.py` runs the final pipeline against a small
set of hand-written messages that are **not** part of the training
dataset, as a quick check that the system generalizes beyond the SMS
Spam Collection data:

```bash
python tests/test_pipeline_sanity.py
```

## Dataset

The [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
dataset: 5,572 SMS messages labeled as `ham` (legitimate) or `spam`.
