import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from typing import List, Tuple


class ToxicityFilter:
    """A *Naïve Bayes*‑based forum‑comment toxicity detector.

    The filter is trained once on the **Wikipedia‑Talk** corpus and can then
    be re‑used to classify arbitrary forum comments as *toxic (1)* or
    *non‑toxic (0)*.

    Attributes
    ----------
    vectorizer : CountVectorizer
        Maps raw text → sparse bag‑of‑words matrix (stop‑words removed,
        lower‑cased).  *Must* be stored so that the **same vocabulary** is
        applied at test time.
    model : MultinomialNB
        The actual probabilistic classifier trained on the vectorised text.
    """

    # ---------------------------------------------------------------------
    # Construction / training
    # ---------------------------------------------------------------------
    def __init__(self, text_train: pd.DataFrame, labels_train: pd.DataFrame) -> None:
        """Fit the vectoriser **and** Naïve Bayes model on the training split.

        Parameters
        ----------
        text_train : pd.DataFrame
            Column of raw comment strings (size *N*).
        labels_train : pd.DataFrame
            Corresponding ground‑truth labels (0 = non‑toxic, 1 = toxic).
        """
        # -----------------------------------------------------------------
        # 1)  Vectorise the raw text → bag‑of‑words counts
        #     ‑ Remove English stop‑words (am, is, the …)
        #     ‑ Lower‑case, strip accents (default), ignore punctuation.
        # -----------------------------------------------------------------
        self.vectorizer: CountVectorizer = CountVectorizer(
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
        )

        # Ensure we feed a *1‑D iterable of str* to scikit‑learn
        train_corpus = text_train.squeeze().astype(str).tolist()
        X_train = self.vectorizer.fit_transform(train_corpus)

        # -----------------------------------------------------------------
        # 2)  Train the Multinomial Naïve Bayes classifier
        # -----------------------------------------------------------------
        y_train = labels_train.squeeze().astype(int).to_numpy()
        self.model: MultinomialNB = MultinomialNB()
        self.model.fit(X_train, y_train)

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    def classify(self, text_test: List[str]) -> List[int]:
        """Classify each comment in *text_test* as toxic or not.

        Notes
        -----
        * **Never** call ``fit``/``fit_transform`` on the vectoriser here – we
          must preserve the original training vocabulary.
        * Returns a Python ``list`` so that the caller can JSON‑serialise or
          further post‑process easily.
        """
        if not text_test:
            return []

        X_test = self.vectorizer.transform(text_test)  # only *transform*!
        predictions = self.model.predict(X_test)
        return predictions.astype(int).tolist()

    # ---------------------------------------------------------------------
    # Convenience wrapper for evaluation (already used by unit tests)
    # ---------------------------------------------------------------------
    def test_model(
        self,
        text_test: pd.DataFrame,
        labels_test: pd.DataFrame,
    ) -> Tuple[str, dict]:
        """Evaluate on a held‑out split and return sklearn's report."""
        preds = self.classify(text_test.squeeze().astype(str).tolist())
        report_str = classification_report(labels_test, preds, output_dict=False)
        report_dict = classification_report(labels_test, preds, output_dict=True)
        return report_str, report_dict
