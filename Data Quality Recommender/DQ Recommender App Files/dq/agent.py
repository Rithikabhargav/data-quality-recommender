"""Preprocessing agent.

The agent works in an observe -> plan -> act -> verify loop over the recommender's issues:

1. observe: audit the current data (profile + issues + quality score);
2. plan:    pick the next action whose target issue is still present;
3. act:     apply it to a copy of the data (respecting the row-loss budget);
4. verify:  re-audit, compare score / target issue / rows, keep the change or roll it back.

Every step is logged with what the agent saw, what it did, why that step matters, and the
measured effect, so the whole run can be explained and reproduced.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .profiler import PLACEHOLDERS, combined_text, is_text_like, missing_masks, normalised_text, profile_dataset
from .recommender import build_issues, grade, quality_scores
from .text_utils import clean_text

MISLABEL_POLICIES = {
    "relabel_confident": "Relabel confident cases, drop the rest",
    "relabel_all": "Relabel every suspected mislabel",
    "drop_all": "Drop every suspected mislabel",
    "flag_only": "Only flag them (no change)",
}


@dataclass
class Step:
    action: str
    title: str
    status: str            # applied | rolled_back | skipped | advice
    observation: str       # what the agent saw before acting
    detail: str            # what it did
    why: str               # how the step helps the dataset / the model
    rows_before: int = 0
    rows_after: int = 0
    cells_changed: int = 0
    score_before: float = 0.0
    score_after: float = 0.0
    target_before: int = 0
    target_after: int = 0
    dims_before: dict = field(default_factory=dict)
    dims_after: dict = field(default_factory=dict)
    code: str = ""

    @property
    def rows_removed(self):
        return self.rows_before - self.rows_after


@dataclass
class Assessment:
    profile: dict
    issues: list
    score: float
    dims: dict

    def affected(self, prefixes):
        return sum(i.affected for i in self.issues if any(i.id.startswith(p) for p in prefixes))


# ---------------------------------------------------------------- helpers
def _dl_for(df, dl):
    """Deep learning results are keyed by original row id (the DataFrame index); map them
    to the current row positions so the recommender can use them."""
    if dl is None:
        return None
    pos = pd.Series(np.arange(len(df)), index=df.index)
    cur = dl[dl["row"].isin(pos.index)].copy()
    cur["row"] = pos.loc[cur["row"]].to_numpy()
    cur.attrs = dict(dl.attrs)
    return cur


def _ids(df, positions):
    positions = [p for p in positions if 0 <= p < len(df)]
    return set(df.index[positions])


def label_value_map(df, label_col, noise):
    """Class name -> the value this dataset uses for it (1..4, 0..3 or names), learned from
    the rows the classifier judged."""
    if noise is None or label_col is None:
        return {}
    given = df[label_col].reindex(noise["row"].to_numpy())
    pairs = pd.DataFrame({"name": noise["given_label"].to_numpy(), "value": given.to_numpy()}).dropna()
    return {n: g["value"].mode().iloc[0] for n, g in pairs.groupby("name")}


# ---------------------------------------------------------------- the agent
class PreprocessingAgent:
    def __init__(self, text_cols, label_col=None, valid_labels=None, vocab=None, label_noise=None,
                 anomalies=None, max_row_loss=0.20, mislabel_policy="relabel_confident",
                 relabel_confidence=0.90, allowed=None, near_dup_rows=3000, keep_snapshots=False):
        self.text_cols = list(text_cols)
        self.label_col = label_col
        self.valid_labels = valid_labels
        self.vocab = vocab
        self.noise = label_noise.copy() if label_noise is not None else None
        self.anomalies = anomalies
        self.max_row_loss = max_row_loss
        self.policy = mislabel_policy
        self.relabel_confidence = relabel_confidence
        self.near_dup_rows = near_dup_rows
        self.allowed = set(allowed) if allowed else None
        self.keep_snapshots = keep_snapshots
        self.snapshots = []   # (label, DataFrame) after each applied step, for step-wise impact
        if self.noise is not None:
            self.noise.attrs = dict(label_noise.attrs)

    # ---- observe
    def assess(self, df):
        prof = profile_dataset(df, self.text_cols, self.label_col, valid_labels=self.valid_labels,
                               vocab=self.vocab, near_dup_rows=self.near_dup_rows)
        issues = build_issues(prof, _dl_for(df, self.noise), _dl_for(df, self.anomalies))
        score, dims = quality_scores(prof, issues)
        return Assessment(prof, issues, score, dims)

    # ---- the action library: (id, title, target issue prefixes, destructive?, why, function)
    def actions(self, columns=()):
        L = self.label_col
        other_cols = [c for c in columns if c not in self.text_cols and c != L]
        return [
            ("standardise_missing", "Standardise disguised missing values", ["disguised_missing"], False,
             "Blank strings and placeholders such as 'N/A' look like data but carry none, and df.isna() does not "
             "see them. Turning them into real NaN gives every later step one consistent definition of 'missing', "
             "so they are counted, imputed or dropped correctly instead of being fed to the model as fake words.",
             self._standardise_missing),
            ("drop_useless_columns", "Drop constant and identifier columns", ["constant_", "id_"], False,
             "A constant column has no information, and an ID column lets a model memorise rows instead of "
             "learning patterns (a common source of leakage). Removing them shrinks the feature space.",
             self._drop_useless_columns),
            ("drop_empty_text", "Drop rows with no text at all", ["empty_text"], True,
             "A row whose every text field is empty gives the model a label with no evidence. It can only teach "
             "the model to guess the majority class.",
             self._drop_empty_text),
            ("drop_invalid_labels", "Drop rows with missing or invalid labels", ["invalid_labels", f"missing_{L}"], True,
             "Supervised training needs a valid target. Values like 0, 5 or 'unknown' would otherwise become "
             "extra fake classes, wasting model capacity and corrupting accuracy metrics.",
             self._drop_invalid_labels),
            ("impute_text", "Fill missing text fields from the other text field", [f"missing_{c}" for c in self.text_cols], False,
             "Throwing away a row because one field is empty also throws away its label and its other field. "
             "Filling the gap from the remaining text keeps the example, removes NaN values that break tokenisers, "
             "and keeps the class balance intact.",
             self._impute_text),
            ("impute_other", "Impute missing numeric / categorical values", [f"missing_{c}" for c in other_cols], False,
             "Most models cannot handle NaN. Median imputation is robust to outliers, the missing-indicator column "
             "keeps the information that the value was absent, and 'Unknown' keeps categorical rows usable.",
             self._impute_other),
            ("repair_text", "Repair encoding and markup artifacts", ["artifacts_"], False,
             "Broken entities like '#39;' and '&lt;b&gt;', tags, URLs and stray backslashes create junk tokens "
             "that split one word into many vocabulary entries (\"nation #39;s\" vs \"nation's\"). Repairing them "
             "shrinks the vocabulary, lowers sparsity and lets the model see the real words.",
             self._repair_text),
            ("resolve_conflicts", "Resolve identical texts with conflicting labels", ["label_conflict"], True,
             "The same text with two labels is a direct contradiction: the model is punished whatever it predicts. "
             "Majority vote keeps one consistent label; groups without a majority are dropped.",
             self._resolve_conflicts),
            ("drop_duplicates", "Remove exact and normalised duplicates", ["dup_rows", "dup_text"], True,
             "Duplicates over-weight some examples and leak between train and test splits, which inflates "
             "validation scores. Keeping one copy gives an honest evaluation.",
             self._drop_duplicates),
            ("drop_near_duplicates", "Remove near-duplicate texts", ["near_dup"], True,
             "Near duplicates (the same story with small edits) cause the same leakage as exact duplicates but "
             "are missed by exact matching.",
             self._drop_near_duplicates),
            ("drop_junk", "Remove garbled / off-domain rows", ["anomalies", "high_oov"], True,
             "Gibberish, heavy typos and non-news content (code, product listings, logs) are noise the model "
             "tries to fit. Removing them reduces vocabulary pollution and the risk of learning spurious patterns.",
             self._drop_junk),
            ("drop_truncated", "Remove truncated / too short texts", ["very_short"], True,
             "A two-word description carries almost no topic signal but still counts as a full training example.",
             self._drop_truncated),
            ("fix_label_noise", "Fix suspected mislabels", ["label_noise"], True,
             "Wrong labels teach the model the wrong answer and make its evaluation unreliable. Rows the classifier "
             "is highly confident about are relabelled to its suggestion; ambiguous ones are removed (policy "
             "configurable).",
             self._fix_label_noise),
            ("clip_outliers", "Clip numeric outliers", ["outliers_"], False,
             "Extreme values dominate means, variances and gradients. Clipping to the 1st-99th percentile keeps "
             "the rows but limits their influence.",
             self._clip_outliers),
            ("log_transform", "Log-transform skewed numeric columns", ["skew_"], False,
             "A log transform makes long-tailed features closer to symmetric, which helps linear models and "
             "neural networks train stably.",
             self._log_transform),
        ]

    # ---- action implementations: return (new_df, detail, cells_changed) or None when nothing to do
    def _standardise_missing(self, df, a):
        out, n = df.copy(), 0
        for c in df.columns:
            if not is_text_like(df[c]):
                continue
            _, blank, ph = missing_masks(df[c])
            mask = blank | ph
            if mask.any():
                out[c] = out[c].astype(object).mask(mask, np.nan)
                n += int(mask.sum())
        return (out, f"Converted {n} blank or placeholder cells to NaN.", n) if n else None

    def _drop_useless_columns(self, df, a):
        cols = [c for c, p in a.profile["columns"].items()
                if (p.get("is_constant") or p.get("looks_like_id")) and c not in self.text_cols and c != self.label_col]
        return (df.drop(columns=cols), f"Dropped columns: {', '.join(map(str, cols))}.", 0) if cols else None

    def _drop_rows(self, df, positions, what):
        ids = _ids(df, positions)
        return (df.drop(index=list(ids)), f"Removed {len(ids)} {what}.", 0) if ids else None

    def _drop_empty_text(self, df, a):
        return self._drop_rows(df, a.profile.get("empty_text_rows", []), "rows with no text")

    def _drop_invalid_labels(self, df, a):
        lab = a.profile.get("label") or {}
        return self._drop_rows(df, lab.get("missing_rows", []) + lab.get("invalid_rows", []),
                               "rows with a missing or invalid label")

    def _impute_text(self, df, a):
        if len(self.text_cols) < 2:
            return None
        out, parts, cells = df.copy(), [], 0
        for c in self.text_cols:
            miss = out[c].isna() | (out[c].astype(object).fillna("").astype(str).str.strip() == "")
            others = combined_text(out, [o for o in self.text_cols if o != c])
            fill = miss & (others.str.len() > 0)
            if not fill.any():
                continue
            # a short field (e.g. a title) gets the first words of the other field, a long one gets it all
            typical = int(np.nanmedian(out.loc[~miss, c].astype(str).str.split().str.len())) if (~miss).any() else 12
            words = max(typical, 6)
            out.loc[fill, c] = others[fill].str.split().str[:words].str.join(" ")
            parts.append(f"{int(fill.sum())} missing '{c}' values (first {words} words of the other field)")
            cells += int(fill.sum())
        if not parts:
            return None
        return out, "Filled " + "; ".join(parts) + ".", cells

    def _impute_other(self, df, a):
        out, parts, n = df.copy(), [], 0
        for c, p in a.profile["columns"].items():
            if p["missing_total"] == 0 or c in self.text_cols or c == self.label_col:
                continue
            miss = out[c].isna()
            if p["kind"] == "numeric":
                out[f"{c}_was_missing"] = miss
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(pd.to_numeric(out[c], errors="coerce").median())
                parts.append(f"'{c}' with the median")
            elif p["kind"] in ("categorical", "boolean"):
                out[c] = out[c].astype(object).where(~miss, "Unknown")
                parts.append(f"'{c}' with 'Unknown'")
            else:
                continue
            n += int(miss.sum())
        return (out, f"Imputed {n} cells: " + ", ".join(parts) + ".", n) if n else None

    def _repair_text(self, df, a):
        out, n = df.copy(), 0
        for c in self.text_cols:
            before = out[c]
            after = before.map(lambda t: clean_text(t) if isinstance(t, str) else t)
            changed = before.notna() & (before.astype(object) != after.astype(object))
            out[c] = after
            n += int(changed.sum())
        return (out, f"Repaired {n} text cells (entities, tags, URLs, backslashes, Unicode).", n) if n else None

    def _resolve_conflicts(self, df, a):
        L = self.label_col
        if L is None:
            return None
        key = normalised_text(combined_text(df, self.text_cols))
        grp = pd.DataFrame({"k": key, "y": df[L].astype(str)}, index=df.index)
        conflicted = grp.groupby("k")["y"].transform("nunique") > 1
        if not conflicted.any():
            return None
        out, fixed, dropped = df.copy(), 0, []
        for k, g in grp[conflicted].groupby("k"):
            counts = g["y"].value_counts()
            if len(counts) > 1 and counts.iloc[0] > counts.iloc[1]:
                majority = df.loc[g.index[g["y"] == counts.index[0]][0], L]
                out.loc[g.index, L] = majority
                fixed += len(g)
            else:
                dropped += list(g.index)
        out = out.drop(index=dropped)
        return out, f"Majority vote fixed {fixed} rows; dropped {len(dropped)} rows in groups without a majority.", fixed

    def _drop_duplicates(self, df, a):
        dup = df.duplicated()
        key = normalised_text(combined_text(df, self.text_cols)) if self.text_cols else None
        if key is not None:
            dup |= key.duplicated() & (key.str.len() > 0)
        ids = df.index[dup]
        return (df.drop(index=ids), f"Removed {len(ids)} duplicate rows (kept the first copy).", 0) if len(ids) else None

    def _drop_near_duplicates(self, df, a):
        pairs = a.profile.get("near_duplicates", {}).get("pairs", [])
        second = {max(p[0], p[1]) for p in pairs}
        return self._drop_rows(df, sorted(second), "near-duplicate rows (kept one of each pair)")

    def _drop_junk(self, df, a):
        ids = set()
        if self.anomalies is not None:
            ids |= set(self.anomalies.loc[self.anomalies["flagged"], "row"]) & set(df.index)
        ids |= _ids(df, (a.profile.get("text_sparsity") or {}).get("high_oov_rows", []))
        return (df.drop(index=list(ids)), f"Removed {len(ids)} garbled or off-domain rows "
                                          "(autoencoder anomalies + rows with over 30% unknown words).", 0) if ids else None

    def _drop_truncated(self, df, a):
        return self._drop_rows(df, a.profile.get("very_short_rows", []), "truncated / very short rows")

    def _fix_label_noise(self, df, a):
        if self.noise is None or self.label_col is None or self.policy == "flag_only":
            return None
        fl = self.noise[self.noise["flagged"] & self.noise["row"].isin(df.index)]
        if fl.empty:
            return None
        values = label_value_map(df, self.label_col, self.noise)
        confident = fl["p_suggested"] >= self.relabel_confidence
        if self.policy == "relabel_all":
            relabel, drop = fl, fl.iloc[0:0]
        elif self.policy == "drop_all":
            relabel, drop = fl.iloc[0:0], fl
        else:
            relabel, drop = fl[confident], fl[~confident]
        relabel = relabel[relabel["suggested_label"].isin(values)]
        out = df.copy()
        if len(relabel):
            out.loc[relabel["row"].to_numpy(), self.label_col] = relabel["suggested_label"].map(values).to_numpy()
        out = out.drop(index=drop["row"].to_numpy())
        # these rows are handled now, so they should not be reported again
        self.noise.loc[self.noise["row"].isin(fl["row"]), "flagged"] = False
        return (out, f"Relabelled {len(relabel)} rows the classifier is at least {self.relabel_confidence:.0%} sure about; "
                     f"dropped {len(drop)} ambiguous ones.", len(relabel))

    def _clip_outliers(self, df, a):
        out, n, cols = df.copy(), 0, []
        for c, p in a.profile["columns"].items():
            if p["kind"] == "numeric" and p.get("outliers_iqr") and c in out:
                x = pd.to_numeric(out[c], errors="coerce")
                lo, hi = x.quantile([0.01, 0.99])
                n += int(((x < lo) | (x > hi)).sum())
                out[c] = x.clip(lo, hi)
                cols.append(str(c))
        return (out, f"Clipped {n} values in {', '.join(cols)} to the 1st-99th percentile.", n) if n else None

    def _log_transform(self, df, a):
        out, cols = df.copy(), []
        for c, p in a.profile["columns"].items():
            if p["kind"] == "numeric" and abs(p.get("skew", 0)) > 2 and p.get("min", -1) >= 0 and c in out:
                out[c] = np.log1p(pd.to_numeric(out[c], errors="coerce"))
                cols.append(str(c))
        return (out, f"Applied log1p to {', '.join(cols)}.", len(out) * len(cols)) if cols else None

    # ---- advice that does not change the data but belongs in the training pipeline
    def advice(self, a):
        tips = []
        p = a.profile
        ts = p.get("text_sparsity") or {}
        if ts:
            max_len = int(np.ceil(p.get("combined_word_p95", 0))) or 64
            tips.append(Step("advice_representation", "Choose vocabulary size and sequence length", "advice",
                             f"The document-term matrix is {ts['sparsity_pct']:.2f}% sparse; {ts['vocab_for_95pct_coverage']:,} "
                             f"words cover 95% of tokens; 95% of rows have at most {max_len} words.",
                             f"Use a vocabulary of about {ts['vocab_for_95pct_coverage']:,} words (min_freq 2), truncate/pad "
                             f"to {max_len} tokens, and use dense embeddings rather than one-hot vectors.",
                             "A vocabulary that covers 95% of tokens keeps nearly all information while dropping the long "
                             "tail of rare words, and a sequence length at the 95th percentile avoids wasting computation on padding.",
                             code=f"MAX_VOCAB = {ts['vocab_for_95pct_coverage']}\nMAX_LEN = {max_len}"))
        lab = p.get("label") or {}
        if lab.get("imbalance_ratio", 1) > 1.5:
            counts = lab["counts"]
            total = sum(counts.values())
            w = {k: round(total / (len(counts) * v), 3) for k, v in counts.items()}
            tips.append(Step("advice_imbalance", "Compensate for class imbalance", "advice",
                             f"Largest/smallest class ratio is {lab['imbalance_ratio']:.2f}.",
                             f"Use class weights {w} in the loss and stratified splits.",
                             "Without weights the model is rewarded for favouring the majority class; weights make every "
                             "class count equally in the loss.", code=f"CLASS_WEIGHTS = {w}"))
        if any(i.id == "near_dup" for i in a.issues):
            tips.append(Step("advice_group_split", "Split by near-duplicate groups", "advice",
                             "Some near duplicates remain.", "Use GroupShuffleSplit with near-duplicate groups as groups.",
                             "Keeps copies of the same story on one side of the split so validation scores stay honest."))
        return tips

    # ---- the loop
    def run(self, df, on_step=None):
        """Runs the agent. Yields nothing; returns (clean_df, steps, first_assessment, last_assessment).
        on_step(step) is called after every decision so a UI can show progress live."""
        df = df.copy()
        n0 = len(df)
        first = current = self.assess(df)
        steps = []
        self.snapshots = [("start", df)] if self.keep_snapshots else []
        for aid, title, targets, destructive, why, fn in self.actions(df.columns):
            if self.allowed is not None and aid not in self.allowed:
                continue
            target_before = current.affected(targets)
            if not targets or target_before == 0:
                continue
            observation = (f"The audit reports {target_before:,} affected " + ("rows" if destructive else "rows/values")
                           + f" for this issue; current score {current.score:.1f}.")
            result = fn(df, current)
            step = Step(aid, title, "skipped", observation, "", why, rows_before=len(df), rows_after=len(df),
                        score_before=current.score, score_after=current.score, target_before=target_before,
                        target_after=target_before, dims_before=dict(current.dims), dims_after=dict(current.dims))
            if result is None:
                step.detail = "Nothing the agent can change automatically for this issue."
                steps.append(step)
                if on_step:
                    on_step(step)
                continue
            new_df, detail, cells = result
            removed_total = n0 - len(new_df)
            if destructive and removed_total > self.max_row_loss * n0:
                step.detail = (f"{detail} Not applied: total row loss would reach {removed_total / n0:.1%}, "
                               f"above the {self.max_row_loss:.0%} budget.")
                steps.append(step)
                if on_step:
                    on_step(step)
                continue
            after = self.assess(new_df)
            step.rows_after, step.cells_changed = len(new_df), cells
            step.score_after, step.dims_after = after.score, dict(after.dims)
            step.target_after = after.affected(targets)
            step.detail = detail
            if after.score + 0.05 < current.score:
                step.status = "rolled_back"
                step.detail += f" Rolled back: the score dropped from {current.score:.1f} to {after.score:.1f}."
                step.rows_after, step.score_after, step.dims_after = len(df), current.score, dict(current.dims)
            else:
                step.status = "applied"
                df, current = new_df, after
                if self.keep_snapshots:
                    self.snapshots.append((title, df))
            steps.append(step)
            if on_step:
                on_step(step)
        for tip in self.advice(current):
            tip.rows_before = tip.rows_after = len(df)
            tip.score_before = tip.score_after = current.score
            steps.append(tip)
            if on_step:
                on_step(tip)
        return df, steps, first, current


# ---------------------------------------------------------------- downstream impact
def score_downstream(d, text_cols, label_col, reference=None, seed=0, repair_inputs=False):
    """Trains a quick TF-IDF + logistic regression model on d and scores it: on a clean
    labelled reference set when one is given, otherwise with 5-fold cross-validation on d.

    repair_inputs=True applies the same text repair to the evaluation texts, as a deployed
    pipeline would do at prediction time after training on repaired data."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    def xy(frame):
        x = combined_text(frame, text_cols)
        y = frame[label_col].astype(object).where(frame[label_col].notna(), "missing").astype(str)
        return x, y.str.replace(r"\.0$", "", regex=True)

    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=1000, C=4.0))
    x, y = xy(d)
    if reference is not None:
        xr, yr = xy(reference)
        if repair_inputs:
            xr = xr.map(clean_text)
        p = model.fit(x, y).predict(xr)
    else:
        keep = y.map(y.value_counts()) >= 5
        x, yr = x[keep], y[keep]
        p = cross_val_predict(model, x, yr, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return {"rows": len(d), "accuracy": accuracy_score(yr, p), "macro_f1": f1_score(yr, p, average="macro")}


def downstream_impact(before, after, text_cols, label_col, reference=None, seed=0, repaired=True):
    """Trains the same quick model on the data before and after the agent and scores both on
    the same evaluation data. With a clean labelled reference set this measures real
    generalisation; without one it falls back to 5-fold cross-validation inside each version
    (then the cleaned data is also easier to predict, so the gain is optimistic)."""
    out = {"before": score_downstream(before, text_cols, label_col, reference, seed),
           "after": score_downstream(after, text_cols, label_col, reference, seed, repair_inputs=repaired)}
    out["method"] = (f"trained on each version, evaluated on {len(reference):,} clean reference rows" if reference is not None
                     else "5-fold cross-validation inside each version (optimistic for the cleaned data)")
    return out


def stepwise_impact(snapshots, text_cols, label_col, reference=None):
    """Downstream accuracy after each applied step (cumulative), to see which steps matter."""
    rows, prev, repaired = [], None, False
    for k, (label, d) in enumerate(snapshots):
        repaired = repaired or label.startswith("Repair")
        r = score_downstream(d, text_cols, label_col, reference, repair_inputs=repaired)
        rows.append({"step": k, "after step": label, "rows": r["rows"], "accuracy": r["accuracy"],
                     "macro_f1": r["macro_f1"], "change": np.nan if prev is None else r["accuracy"] - prev})
        prev = r["accuracy"]
    return pd.DataFrame(rows)


def steps_frame(steps):
    rows = []
    for s in steps:
        d = asdict(s)
        d.pop("dims_before"); d.pop("dims_after")
        d["rows_removed"] = s.rows_removed
        d["score_change"] = round(s.score_after - s.score_before, 2)
        rows.append(d)
    return pd.DataFrame(rows)


def agent_script(steps, text_cols, label_col):
    """A plain pandas script that records what the agent did (for the report / reproducibility)."""
    lines = ["# Preprocessing steps applied by the DQ-Recommender agent", "import numpy as np", "import pandas as pd",
             "from dq.text_utils import clean_text", "", f"TEXT_COLS = {text_cols!r}", f"LABEL_COL = {label_col!r}",
             "df = pd.read_csv('your_data.csv')", ""]
    for k, s in enumerate([s for s in steps if s.status in ("applied", "advice")], 1):
        lines.append(f"# {k}. {s.title} [{s.status}]")
        lines.append(f"#    {s.detail}")
        if s.code:
            lines.append(s.code)
        lines.append("")
    return "\n".join(lines)


def agent_markdown(steps, first, last, impact=None):
    out = ["# Preprocessing agent log", "",
           f"Score: {first.score:.1f} ({grade(first.score)}) -> {last.score:.1f} ({grade(last.score)})", ""]
    if impact:
        b, a = impact["before"], impact["after"]
        out += [f"Downstream model ({impact['method']}): accuracy {b['accuracy']:.4f} -> {a['accuracy']:.4f}, "
                f"macro-F1 {b['macro_f1']:.4f} -> {a['macro_f1']:.4f}", ""]
    for k, s in enumerate(steps, 1):
        out += [f"## {k}. {s.title} [{s.status}]", f"- Observed: {s.observation}", f"- Action: {s.detail}",
                f"- Why it helps: {s.why}"]
        if s.status == "applied":
            out.append(f"- Effect: rows {s.rows_before:,} -> {s.rows_after:,}, score {s.score_before:.1f} -> "
                       f"{s.score_after:.1f}, affected {s.target_before:,} -> {s.target_after:,}")
        out.append("")
    return "\n".join(out)
