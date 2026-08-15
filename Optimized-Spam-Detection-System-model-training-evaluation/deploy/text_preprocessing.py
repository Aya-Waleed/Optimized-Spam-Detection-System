"""
Shared text preprocessing for the Spam Detection System.

This module is imported both by the deployment pipeline (when it is
built and saved in notebooks/Model_Optimization_Deployment.ipynb) and by
the deployment scripts (deploy/predict_cli.py, deploy/app.py). Keeping
this logic in a single, importable module (instead of copy-pasting it or
defining it inline in a notebook) is required for the saved pipeline to
un-pickle correctly, since a pickled function is referenced by its
module path.

The preprocessing steps mirror exactly what was used to build the
training data in notebooks/Spam_Detection.ipynb:
    lowercase -> remove punctuation -> tokenize -> remove stopwords is
    NOT applied here (stopwords were kept in the original pipeline) ->
    stem
"""

import re
import string

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def _ensure_nltk_data():
    for resource, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()
_stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    """Apply the same cleaning/stemming pipeline used during training."""
    text = text.lower()

    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.strip()]
    tokens = [_stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


def preprocess_messages(messages):
    """Vectorized-friendly wrapper: preprocess a list of raw messages.

    This is the function referenced by the FunctionTransformer inside
    the saved deployment pipeline, so raw text passed to the pipeline is
    cleaned and stemmed exactly like the training data before TF-IDF.
    """
    return [preprocess_text(str(message)) for message in messages]
