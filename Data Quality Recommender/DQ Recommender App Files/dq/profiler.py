"""Statistical data quality profiler.

profile_dataset() returns a nested dict describing completeness, duplicates, sparsity,
text artifacts, outliers, label health and correlations. The recommender turns this
dict (plus the deep learning checks) into issues and preprocessing recommendations.
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .text_utils import ARTIFACT_PATTERNS, tokenize

PLACEHOLDERS = {"nan", "null", "none", "n/a", "na", "-", "--", "?", "missing", "undefined", "#n/a"}
LABEL_NAME_HINTS = re.compile(r"(label|class|target|category|topic|y$)", re.I)
NORMALISE = r"[^a-z0-9 ]+"


def rx_contains(s, rx):
    """Python-regex version of str.contains (pyarrow's RE2 has no look-behind or back-references)."""
    return s.map(lambda t: isinstance(t, str) and rx.search(t) is not None).astype(bool)


# ---------------------------------------------------------------- column typing
def is_text_like(s):
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s))


def detect_text_columns(df):
    cols = []
    for c in df.columns:
        s = df[c]
        if not is_text_like(s):
            continue
        vals = s.dropna().astype(str)
        if len(vals) == 0:
            continue
        if vals.str.len().median() >= 25 and vals.nunique() / len(vals) > 0.3:
            cols.append(c)
    return cols


def detect_label_column(df, text_cols):
    n = len(df)
    candidates = []
    for c in df.columns:
        if c in text_cols:
            continue
        nun = df[c].nunique(dropna=True)
        if 2 <= nun <= 50 and nun <= max(2, 0.05 * n):
            candidates.append((0 if LABEL_NAME_HINTS.search(str(c)) else 1, nun, c))
    return sorted(candidates)[0][2] if candidates else None


def column_kind(s, text_cols, label_col):
    if s.name == label_col:
        return "label"
    if s.name in text_cols:
        return "text"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "categorical"


# ---------------------------------------------------------------- missing values
def missing_masks(s):
    """Return (nan_mask, blank_mask, placeholder_mask) for a column."""
    nan = s.isna()
    if is_text_like(s):
        as_str = s.astype(str).where(~nan, "")
        stripped = as_str.str.strip()
        blank = (~nan) & (stripped == "")
        placeholder = (~nan) & (~blank) & stripped.str.lower().isin(PLACEHOLDERS)
    else:
        blank = pd.Series(False, index=s.index)
        placeholder = pd.Series(False, index=s.index)
    return nan, blank, placeholder


def any_missing_mask(df):
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        nan, blank, placeholder = missing_masks(df[c])
        mask |= nan | blank | placeholder
    return mask


def iqr_bounds(x, k=1.5):
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


# ---------------------------------------------------------------- per column
def profile_column(s, kind):
    n = len(s)
    nan, blank, placeholder = missing_masks(s)
    missing = nan | blank | placeholder
    p = {
        "name": str(s.name), "kind": kind, "dtype": str(s.dtype),
        "missing_nan": int(nan.sum()), "missing_blank": int(blank.sum()),
        "missing_placeholder": int(placeholder.sum()), "missing_total": int(missing.sum()),
        "missing_pct": float(missing.mean() * 100) if n else 0.0,
        "n_unique": int(s[~missing].nunique()),
    }
    valid = s[~missing]
    p["unique_ratio"] = float(p["n_unique"] / max(len(valid), 1))
    p["is_constant"] = p["n_unique"] <= 1 and len(valid) > 0

    if kind == "numeric":
        x = pd.to_numeric(valid, errors="coerce").dropna().astype(float)
        if len(x):
            lo, hi = iqr_bounds(x)
            p.update(mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()),
                     zeros_pct=float((x == 0).mean() * 100), skew=float(x.skew()) if len(x) > 2 else 0.0,
                     outliers_iqr=int(((x < lo) | (x > hi)).sum()))
    elif kind == "text":
        txt = valid.astype(str)
        chars = txt.str.len()
        words = txt.str.split().str.len()
        lo, hi = iqr_bounds(chars) if len(chars) else (0, 0)
        letters = txt.str.count(r"[A-Za-z]")
        upper = txt.str.count(r"[A-Z]")
        p.update(
            char_len_mean=float(chars.mean()) if len(chars) else 0.0,
            char_len_median=float(chars.median()) if len(chars) else 0.0,
            char_len_p95=float(chars.quantile(0.95)) if len(chars) else 0.0,
            word_len_mean=float(words.mean()) if len(words) else 0.0,
            word_len_p95=float(words.quantile(0.95)) if len(words) else 0.0,
            very_short=int((words < 3).sum()),
            length_outliers_low=int((chars < lo).sum()),
            length_outliers_high=int((chars > hi).sum()),
            shouting=int(((upper / letters.clip(lower=1)) > 0.7).where(letters > 10, False).sum()),
            duplicated_values=int(txt.duplicated().sum()),
            artifacts={k: int(rx_contains(txt, rx).sum()) for k, rx in ARTIFACT_PATTERNS.items()},
        )
    elif kind in ("categorical", "label"):
        vc = valid.astype(str).value_counts()
        p["top_values"] = {str(k): int(v) for k, v in vc.head(10).items()}
        p["looks_like_id"] = kind == "categorical" and p["unique_ratio"] > 0.95 and len(valid) > 50
    return p


# ---------------------------------------------------------------- dataset level
def combined_text(df, text_cols):
    if not text_cols:
        return pd.Series("", index=df.index)
    parts = [df[c].where(df[c].notna(), "").astype(str) for c in text_cols]
    out = parts[0]
    for part in parts[1:]:
        out = out + " " + part
    return out.str.strip()


def normalised_text(texts):
    return texts.str.lower().str.replace(NORMALISE, " ", regex=True).str.split().str.join(" ")


def text_sparsity(texts, vocab=None):
    """Bag-of-words sparsity, vocabulary size and coverage statistics."""
    texts = texts[texts.str.len() > 0]
    if len(texts) == 0:
        return {}
    cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    X = cv.fit_transform(texts)
    n, v = X.shape
    freqs = np.asarray(X.sum(axis=0)).ravel()
    order = np.sort(freqs)[::-1]
    cum = np.cumsum(order) / order.sum()
    out = {
        "rows": int(n), "vocab_size": int(v), "nnz": int(X.nnz),
        "sparsity_pct": float(100 * (1 - X.nnz / (n * v))),
        "avg_unique_tokens_per_row": float(X.nnz / n),
        "hapax_pct": float(100 * (freqs == 1).mean()),
        "vocab_for_90pct_coverage": int(np.searchsorted(cum, 0.90) + 1),
        "vocab_for_95pct_coverage": int(np.searchsorted(cum, 0.95) + 1),
        "coverage_curve": [(int(k), float(cum[k - 1])) for k in
                           sorted(set(np.unique(np.geomspace(1, v, 60).astype(int))))],
        "top_tokens": {w: int(c) for w, c in sorted(zip(cv.get_feature_names_out(), freqs),
                                                     key=lambda t: -t[1])[:25]},
    }
    if vocab is not None:
        oov = texts.map(lambda t: vocab.oov_rate(tokenize(t)))
        out["mean_oov_pct"] = float(oov.mean() * 100)
        out["high_oov_rows"] = [int(i) for i in oov.index[oov > 0.3]]
    return out


def near_duplicates(texts, max_rows=5000, threshold=0.9, seed=0):
    """Pairs of different texts with TF-IDF cosine similarity >= threshold (on a sample)."""
    texts = texts[texts.str.len() > 0].drop_duplicates()
    if len(texts) < 3:
        return {"checked_rows": int(len(texts)), "pairs": []}
    if len(texts) > max_rows:
        texts = texts.sample(max_rows, random_state=seed)
    X = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None).fit_transform(texts)
    nn_ = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X)
    dist, idx = nn_.kneighbors(X)
    pairs, seen = [], set()
    for i in range(len(texts)):
        j, sim = idx[i, 1], 1 - dist[i, 1]
        key = tuple(sorted((i, j)))
        if sim >= threshold and key not in seen:
            seen.add(key)
            pairs.append((int(texts.index[i]), int(texts.index[j]), float(sim)))
    return {"checked_rows": int(len(texts)), "pairs": pairs}


def label_profile(labels, valid_labels=None):
    nan, blank, placeholder = missing_masks(labels)
    missing = nan | blank | placeholder
    vals = labels[~missing].astype(str)
    out = {"missing": int(missing.sum()), "missing_rows": [int(i) for i in labels.index[missing]]}
    if valid_labels is not None:
        allowed = {str(v) for v in valid_labels}
        bad = ~vals.isin(allowed)
        out["invalid"] = int(bad.sum())
        out["invalid_rows"] = [int(i) for i in vals.index[bad]]
        out["invalid_values"] = sorted(vals[bad].unique().tolist())[:20]
        vals = vals[~bad]
    counts = vals.value_counts()
    out["counts"] = {str(k): int(v) for k, v in counts.items()}
    if len(counts) >= 2:
        p = counts / counts.sum()
        out["imbalance_ratio"] = float(counts.max() / counts.min())
        out["norm_entropy"] = float(-(p * np.log(p)).sum() / np.log(len(p)))
        out["rare_classes"] = [str(k) for k, v in p.items() if v < 0.01]
    else:
        out.update(imbalance_ratio=1.0, norm_entropy=1.0, rare_classes=[])
    return out


def engineered_features(df, text_cols):
    """Numeric view of text columns so correlations can be computed for text data too."""
    feats = pd.DataFrame(index=df.index)
    for c in text_cols:
        s = df[c].where(df[c].notna(), "").astype(str)
        feats[f"{c}__chars"] = s.str.len()
        feats[f"{c}__words"] = s.str.split().str.len()
        feats[f"{c}__digit_ratio"] = s.str.count(r"\d") / s.str.len().clip(lower=1)
        feats[f"{c}__upper_ratio"] = s.str.count(r"[A-Z]") / s.str.len().clip(lower=1)
        feats[f"{c}__punct_ratio"] = s.str.count(r"[^\w\s]") / s.str.len().clip(lower=1)
    return feats


def correlations(df, text_cols, threshold=0.95):
    num = df.select_dtypes(include="number")
    frame = pd.concat([num, engineered_features(df, text_cols)], axis=1)
    frame = frame.loc[:, frame.nunique() > 1]
    if frame.shape[1] < 2:
        return {"matrix": None, "high_pairs": []}
    corr = frame.corr(numeric_only=True)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append((cols[i], cols[j], float(r)))
    return {"matrix": corr, "high_pairs": pairs}


# ---------------------------------------------------------------- main entry point
def profile_dataset(df, text_cols=None, label_col=None, valid_labels=None, vocab=None,
                    near_dup_rows=5000):
    df = df.reset_index(drop=True)
    text_cols = [c for c in (text_cols if text_cols is not None else detect_text_columns(df)) if c in df]
    if label_col is None:
        label_col = detect_label_column(df, text_cols)
    elif label_col not in df:
        label_col = None

    n_rows, n_cols = df.shape
    cols = {c: profile_column(df[c], column_kind(df[c], text_cols, label_col)) for c in df.columns}
    missing_cells = sum(p["missing_total"] for p in cols.values())

    text = combined_text(df, text_cols)
    norm = normalised_text(text)
    has_text = norm.str.len() > 0
    dup_text_mask = norm.duplicated(keep=False) & has_text

    result = {
        "n_rows": int(n_rows), "n_cols": int(n_cols),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
        "text_cols": list(text_cols), "label_col": label_col,
        "columns": cols,
        "missing_cells": int(missing_cells),
        "missing_cells_pct": float(100 * missing_cells / max(n_rows * n_cols, 1)),
        "rows_with_missing": int(any_missing_mask(df).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_text_rows": int((norm.duplicated() & has_text).sum()),
        "empty_text_rows": [int(i) for i in norm.index[~has_text]] if text_cols else [],
    }

    if text_cols:
        # a duplicate text only counts as a conflict if the copies disagree on the label
        conflicts = []
        if label_col is not None:
            grp = pd.DataFrame({"t": norm[dup_text_mask], "y": df.loc[dup_text_mask, label_col].astype(str)})
            n_labels = grp.groupby("t")["y"].nunique()
            bad_texts = set(n_labels.index[n_labels > 1])
            conflicts = [int(i) for i in grp.index[grp["t"].isin(bad_texts)]]
        result["conflicting_label_rows"] = conflicts
        words = text.str.split().str.len()
        result["combined_word_p95"] = float(words[has_text].quantile(0.95)) if has_text.any() else 0.0
        # a long-form column (median >= 10 words) holding < 3 words looks truncated
        short = (words < 3) & has_text
        for c in text_cols:
            if cols[c].get("word_len_mean", 0) >= 10:
                w = df[c].where(df[c].notna(), "").astype(str).str.split().str.len()
                short |= (w > 0) & (w < 3)
        result["very_short_rows"] = [int(i) for i in short.index[short]]
        result["text_sparsity"] = text_sparsity(text, vocab)
        result["near_duplicates"] = near_duplicates(text, max_rows=near_dup_rows)
        artifact_rows = pd.Series(False, index=df.index)
        for c in text_cols:
            s = df[c].where(df[c].notna(), "").astype(str)
            for rx in ARTIFACT_PATTERNS.values():
                artifact_rows |= rx_contains(s, rx)
        result["artifact_rows"] = int(artifact_rows.sum())

    numeric_cols = [c for c, p in cols.items() if p["kind"] == "numeric"]
    if numeric_cols:
        num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        result["numeric_zero_sparsity_pct"] = float(100 * ((num == 0) | num.isna()).to_numpy().mean())

    if label_col is not None:
        result["label"] = label_profile(df[label_col], valid_labels)

    result["correlation"] = correlations(df, text_cols)
    return result
