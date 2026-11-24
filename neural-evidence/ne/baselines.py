"""E1 traditional baselines: majority class and TF-IDF + logistic regression."""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import metrics
from .data import evidence_of


def majority(train_rows):
    return int(np.bincount([r["label"] for r in train_rows]).argmax())


class TfidfLR:
    """Features: TF-IDF(claim) | TF-IDF(evidence) | TF-IDF(claim) * TF-IDF(evidence).

    The element-wise product (shared vocabulary) lets a linear model see lexical
    overlap between claim and evidence. evidence='none' gives a claim-only model.
    """

    def __init__(self, evidence="gold", C=1.0, max_features=200_000):
        self.evidence, self.C = evidence, C
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features,
                                   sublinear_tf=True, strip_accents="unicode")

    def _features(self, rows):
        c = self.vec.transform([r["claim"] for r in rows])
        if self.evidence == "none":
            return c
        e = self.vec.transform([evidence_of(r, self.evidence) for r in rows])
        return sp.hstack([c, e, c.multiply(e)]).tocsr()

    def fit(self, rows):
        texts = [r["claim"] for r in rows]
        if self.evidence != "none":
            texts += [evidence_of(r, self.evidence) for r in rows]
        self.vec.fit(texts)
        self.clf = LogisticRegression(C=self.C, max_iter=2000)
        self.clf.fit(self._features(rows), [r["label"] for r in rows])
        return self

    def predict(self, rows):
        return self.clf.predict(self._features(rows))


def tune_tfidf(train_rows, val_rows, evidence="gold", Cs=(0.3, 1.0, 3.0, 10.0)):
    """Pick C on validation macro-F1; returns (best model, {C: val macro-F1})."""
    scores, best = {}, None
    for C in Cs:
        m = TfidfLR(evidence, C).fit(train_rows)
        scores[C] = metrics.evaluate([r["label"] for r in val_rows], m.predict(val_rows))["macro_f1"]
        print(f"  C={C}: val macro-F1 {scores[C]:.4f}")
        if best is None or scores[C] > scores[best[0]]:
            best = (C, m)
    return best[1], scores
