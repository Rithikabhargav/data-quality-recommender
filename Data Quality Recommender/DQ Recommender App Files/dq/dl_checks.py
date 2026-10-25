"""Deep learning based quality checks: label noise and content anomalies."""
import json
import os

import numpy as np
import pandas as pd
import torch

from .models import CharAutoencoder, NewsClassifier, reconstruction_error
from .text_utils import Vocab, char_ngram_matrix, clean_text, tokenize

LABEL_ALIASES = {
    "world": 0, "sports": 1, "sport": 1, "business": 2,
    "sci/tech": 3, "sci-tech": 3, "scitech": 3, "science": 3, "technology": 3, "tech": 3,
}


def length_adjusted_score(err, nnz, cal):
    """Robust z-score of log reconstruction error within the text-length bin.

    Short texts have few active n-grams, each carrying a large weight after L2
    normalisation, so their raw error is naturally higher. Comparing each row with clean
    rows of similar length removes that bias.
    """
    edges, med, mad = (np.asarray(cal[k]) for k in ("edges", "median", "mad"))
    k = np.clip(np.searchsorted(edges, nnz, side="right") - 1, 0, len(med) - 1)
    return (np.log(np.maximum(err, 1e-12)) - med[k]) / mad[k]


class DLChecker:
    """Loads the trained artifacts and runs both deep learning checks."""

    def __init__(self, artifact_dir="artifacts", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(artifact_dir, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        self.vocab = Vocab.load(os.path.join(artifact_dir, "vocab.json"))
        self.class_names = self.config["class_names"]
        self.max_len = self.config["max_len"]

        self.classifier = NewsClassifier(**self.config["classifier"]).to(self.device)
        self.classifier.load_state_dict(torch.load(os.path.join(artifact_dir, "classifier.pt"),
                                                   map_location=self.device))
        self.classifier.eval()

        self.autoencoder = CharAutoencoder(**self.config["autoencoder"]).to(self.device)
        self.autoencoder.load_state_dict(torch.load(os.path.join(artifact_dir, "autoencoder.pt"),
                                                    map_location=self.device))
        self.autoencoder.eval()
        self.ae_cal = self.config["ae_calibration"]
        self.ae_threshold = self.ae_cal["threshold"]
        self.ae_reference = np.array(self.ae_cal["reference_quantiles"])

    # ------------------------------------------------------------------ labels
    def map_labels(self, labels):
        """Map user labels onto the 4 AG News classes. Returns (index array, ok mask) or None."""
        s = labels.astype(str).str.strip().str.lower().str.replace(r"\.0$", "", regex=True)
        s = s[labels.notna()].reindex(labels.index)
        present = s.dropna()
        if present.empty:
            return None
        # decide the encoding from the majority of values, so a few bad labels
        # ("unknown", 0, 5) do not switch the whole check off
        share = lambda allowed: present.isin(allowed).mean()
        if share({"1", "2", "3", "4"}) >= 0.9 and not (present == "0").mean() > 0.05:
            idx = pd.to_numeric(s.where(s.isin({"1", "2", "3", "4"})), errors="coerce") - 1
        elif share({"0", "1", "2", "3"}) >= 0.9:
            idx = pd.to_numeric(s.where(s.isin({"0", "1", "2", "3"})), errors="coerce")
        elif share(set(LABEL_ALIASES)) >= 0.5:
            idx = s.map(LABEL_ALIASES)
        else:
            return None
        ok = idx.notna() & idx.isin([0, 1, 2, 3])
        return idx.fillna(-1).astype(int).to_numpy(), ok.to_numpy()

    def predict_proba(self, texts, batch_size=512):
        cleaned = [clean_text(t) for t in texts]
        probs = []
        with torch.no_grad():
            for i in range(0, len(cleaned), batch_size):
                x = torch.from_numpy(self.vocab.encode_batch(cleaned[i:i + batch_size], self.max_len))
                probs.append(torch.softmax(self.classifier(x.to(self.device)), dim=1).cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros((0, len(self.class_names)))

    def label_noise(self, texts, labels, min_tokens=12):
        """Confident-learning style check.

        For each class j the threshold t_j is the average probability the model gives to
        class j on rows labelled j. A row labelled y is suspicious when some other class k
        gets p_k >= t_k, i.e. the model is as sure about k as it usually is about real k rows.
        """
        mapped = self.map_labels(pd.Series(labels))
        if mapped is None:
            return None
        y, ok = mapped
        texts = pd.Series(texts).reset_index(drop=True).fillna("").astype(str)
        # too little text to judge a topic (e.g. title only): leave those to the other checks
        n_tokens = texts.map(lambda t: len(tokenize(clean_text(t)))).to_numpy()
        rows = np.where(ok & (n_tokens >= min_tokens))[0]
        if len(rows) == 0:
            return None
        P = self.predict_proba(texts.iloc[rows].tolist())
        yg = y[rows]
        k = len(self.class_names)
        thresholds = np.array([P[yg == j, j].mean() if (yg == j).any() else 1.0 for j in range(k)])
        P_other = P.copy()
        P_other[np.arange(len(rows)), yg] = -1
        above = P_other >= thresholds[None, :]
        pred = np.where(above.any(1), np.where(above, P_other, -1).argmax(1), P.argmax(1))
        flagged = above.any(1)
        out = pd.DataFrame({
            "row": rows,
            "given_label": [self.class_names[j] for j in yg],
            "predicted_label": [self.class_names[j] for j in P.argmax(1)],
            "suggested_label": [self.class_names[j] for j in pred],
            "p_given": P[np.arange(len(rows)), yg],
            "p_suggested": P[np.arange(len(rows)), pred],
            "flagged": flagged,
        })
        out["margin"] = out["p_suggested"] - out["p_given"]
        out.attrs["thresholds"] = {c: round(float(t), 3) for c, t in zip(self.class_names, thresholds)}
        out.attrs["accuracy_vs_given"] = float((P.argmax(1) == yg).mean())
        out.attrs["skipped_short"] = int((ok & (n_tokens < min_tokens)).sum())
        return out

    # ------------------------------------------------------------------ anomalies
    def anomaly_scores(self, texts, batch_size=1024):
        """Raw reconstruction error, number of active n-grams and length-adjusted z-score."""
        cleaned = [clean_text(t) for t in texts]
        X = char_ngram_matrix(cleaned)
        errs = []
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i:i + batch_size].toarray()).to(self.device)
            errs.append(reconstruction_error(self.autoencoder, xb).cpu().numpy())
        err = np.concatenate(errs) if errs else np.zeros(0)
        nnz = np.diff(X.indptr)
        return err, nnz, length_adjusted_score(err, nnz, self.ae_cal)

    def anomalies(self, texts):
        texts = pd.Series(texts).reset_index(drop=True).fillna("").astype(str)
        has_text = texts.str.strip().str.len() > 0
        rows = np.where(has_text.to_numpy())[0]
        if len(rows) == 0:
            return None
        err, nnz, z = self.anomaly_scores(texts.iloc[rows].tolist())
        pct = np.searchsorted(self.ae_reference, z) / max(len(self.ae_reference) - 1, 1) * 100
        return pd.DataFrame({
            "row": rows, "recon_error": err, "n_ngrams": nnz, "anomaly_z": z,
            "reference_percentile": np.clip(pct, 0, 100), "flagged": z > self.ae_threshold,
        })
