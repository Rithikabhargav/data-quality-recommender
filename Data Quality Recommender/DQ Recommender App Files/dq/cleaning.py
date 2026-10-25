"""Applies the recommended preprocessing steps and logs what changed."""
import numpy as np
import pandas as pd

from .profiler import PLACEHOLDERS, combined_text, normalised_text
from .text_utils import clean_text

DEFAULT_STEPS = ("standardise_missing", "repair_text", "drop_empty_text", "drop_invalid_labels",
                 "drop_duplicates", "drop_conflicts", "drop_short", "drop_anomalies", "fix_label_noise")


def apply_cleaning(df, profile, steps=DEFAULT_STEPS, label_noise=None, anomalies=None,
                   valid_labels=None, label_noise_action="drop"):
    df = df.reset_index(drop=True).copy()
    text_cols, label_col = profile["text_cols"], profile.get("label_col")
    log = []

    def note(step, before):
        log.append({"step": step, "rows_before": before, "rows_after": len(df), "removed": before - len(df)})

    if "standardise_missing" in steps:
        for c in df.columns:
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                s = df[c].astype("object")
                stripped = s.where(s.isna(), s.astype(str).str.strip())
                bad = stripped.isin([""]) | stripped.astype(str).str.lower().isin(PLACEHOLDERS)
                df[c] = s.mask(bad & s.notna(), np.nan)
        note("standardise_missing", len(df))

    # row-level decisions from the DL checks refer to the original row positions
    drop_rows = set()
    if "drop_anomalies" in steps:
        if anomalies is not None:
            drop_rows |= set(anomalies.loc[anomalies["flagged"], "row"].tolist())
        drop_rows |= set((profile.get("text_sparsity") or {}).get("high_oov_rows", []))
    if "drop_short" in steps:
        drop_rows |= set(profile.get("very_short_rows", []))
    if "fix_label_noise" in steps and label_noise is not None and label_col:
        flagged = label_noise[label_noise["flagged"]]
        if label_noise_action == "relabel":
            # write the suggestion in the dataset's own label encoding (1-4, 0-3 or names)
            from .agent import label_value_map
            values = label_value_map(df, label_col, label_noise)
            for r, lab in zip(flagged["row"], flagged["suggested_label"]):
                if lab in values:
                    df.at[r, label_col] = values[lab]
            log.append({"step": "relabel_label_noise", "rows_before": len(df), "rows_after": len(df),
                        "removed": 0, "relabelled": len(flagged)})
        else:
            drop_rows |= set(flagged["row"].tolist())
    if drop_rows:
        before = len(df)
        df = df.drop(index=[r for r in drop_rows if r in df.index])
        note("drop_dl_flagged_rows", before)

    if "repair_text" in steps:
        for c in text_cols:
            df[c] = df[c].map(lambda t: clean_text(t) if isinstance(t, str) else t)
        note("repair_text", len(df))

    text = combined_text(df, text_cols)
    if "drop_empty_text" in steps and text_cols:
        before = len(df)
        df = df[text.str.strip().str.len() > 0]
        note("drop_empty_text", before)

    if "drop_invalid_labels" in steps and label_col:
        before = len(df)
        keep = df[label_col].notna()
        if valid_labels is not None:
            keep &= df[label_col].astype(str).isin({str(v) for v in valid_labels})
        df = df[keep]
        note("drop_invalid_or_missing_labels", before)

    if "drop_conflicts" in steps and label_col and text_cols:
        before = len(df)
        key = normalised_text(combined_text(df, text_cols))
        n_lab = df.groupby(key)[label_col].transform("nunique")
        df = df[n_lab <= 1]
        note("drop_conflicting_labels", before)

    if "drop_duplicates" in steps:
        before = len(df)
        df = df.drop_duplicates()
        if text_cols:
            key = normalised_text(combined_text(df, text_cols))
            df = df[~key.duplicated()]
        note("drop_duplicates", before)

    if "drop_short" in steps and text_cols:
        # rows that are short only after repairing the text
        before = len(df)
        df = df[combined_text(df, text_cols).str.split().str.len() >= 3]
        note("drop_short_text", before)

    if text_cols:
        for c in text_cols:
            df[c] = df[c].fillna("")
    return df.reset_index(drop=True), pd.DataFrame(log)
