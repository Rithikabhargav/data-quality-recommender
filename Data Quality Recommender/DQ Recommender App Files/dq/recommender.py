"""Turns profiler + deep learning findings into scored issues and preprocessing advice."""
import html as htmllib
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
SEVERITY_COLOR = {"critical": "#b42318", "high": "#c4320a", "medium": "#b54708", "low": "#475467", "info": "#1d4ed8"}


@dataclass
class Issue:
    id: str
    category: str
    title: str
    severity: str
    affected: int
    pct: float
    detail: str
    recommendation: str
    code: str = ""
    step: int = 99          # position in the recommended preprocessing pipeline
    rows: list = field(default_factory=list)


def severity_from_pct(pct, critical=20, high=5, medium=1):
    if pct >= critical:
        return "critical"
    if pct >= high:
        return "high"
    if pct >= medium:
        return "medium"
    return "low"


def _pct(k, n):
    return 100.0 * k / max(n, 1)


# ---------------------------------------------------------------- rules
def build_issues(profile, label_noise=None, anomalies=None):
    n = profile["n_rows"]
    issues = []
    add = issues.append
    text_cols = profile["text_cols"]
    label_col = profile.get("label_col")

    # 1. completeness -------------------------------------------------------
    for name, c in profile["columns"].items():
        if c["missing_total"] == 0:
            continue
        pct = c["missing_pct"]
        parts = []
        if c["missing_nan"]:
            parts.append(f"{c['missing_nan']} NaN")
        if c["missing_blank"]:
            parts.append(f"{c['missing_blank']} empty/whitespace strings")
        if c["missing_placeholder"]:
            parts.append(f"{c['missing_placeholder']} placeholder tokens (e.g. 'n/a', 'null')")
        if c["kind"] == "text":
            others = [t for t in text_cols if t != name]
            rec = (f"Text cannot be imputed meaningfully. Fill missing '{name}' with an empty string and "
                   f"rely on {', '.join(repr(o) for o in others)} for those rows; drop rows where every text field is empty."
                   if others else f"Drop rows where '{name}' is missing; it is the only model input.")
            code = f"df['{name}'] = df['{name}'].replace(r'^\\s*$', np.nan, regex=True).fillna('')"
        elif c["kind"] == "numeric":
            rec = ("Impute with the median (robust to outliers) and add a missing-indicator column; "
                   "use KNN/iterative imputation if the column is important." if pct < 40 else
                   "More than 40% missing: consider dropping the column or modelling missingness explicitly.")
            code = f"df['{name}_was_missing'] = df['{name}'].isna()\ndf['{name}'] = df['{name}'].fillna(df['{name}'].median())"
        elif c["kind"] == "label":
            rec = "Rows without a label cannot be used for supervised training: drop them or send them for annotation."
            code = f"df = df.dropna(subset=['{label_col}'])"
        else:
            rec = "Impute with the mode or an explicit 'Unknown' category."
            code = f"df['{name}'] = df['{name}'].fillna('Unknown')"
        add(Issue(f"missing_{name}", "Completeness", f"Missing values in '{name}'",
                  severity_from_pct(pct, 30, 5, 0.5), c["missing_total"], pct,
                  "; ".join(parts) + ".", rec, code, step=1))
        if c["missing_blank"] or c["missing_placeholder"]:
            add(Issue(f"disguised_missing_{name}", "Completeness", f"Disguised missing values in '{name}'",
                      "medium", c["missing_blank"] + c["missing_placeholder"],
                      _pct(c["missing_blank"] + c["missing_placeholder"], n),
                      "Empty strings or placeholder tokens are not counted by df.isna(), so they slip past naive checks.",
                      "Convert blanks and placeholder tokens to real NaN before any other step.",
                      "df = df.replace(r'^\\s*$', np.nan, regex=True)\n"
                      "df = df.replace({'n/a': np.nan, 'null': np.nan, 'none': np.nan, '?': np.nan})", step=0))

    if profile.get("empty_text_rows"):
        k = len(profile["empty_text_rows"])
        add(Issue("empty_text", "Completeness", "Rows with no usable text", severity_from_pct(_pct(k, n), 10, 2, 0.2),
                  k, _pct(k, n), "Every text column is empty for these rows, so they carry no signal.",
                  "Drop these rows.", "df = df[df[TEXT_COLS].fillna('').apply(' '.join, axis=1).str.strip() != '']",
                  step=2, rows=profile["empty_text_rows"][:500]))

    # 2. uniqueness ---------------------------------------------------------
    if profile["duplicate_rows"]:
        k = profile["duplicate_rows"]
        add(Issue("dup_rows", "Uniqueness", "Exact duplicate rows", severity_from_pct(_pct(k, n), 10, 2, 0.2), k, _pct(k, n),
                  "Identical rows inflate the weight of some examples and leak between train and test splits.",
                  "Drop exact duplicates, keeping the first occurrence.", "df = df.drop_duplicates()", step=4))
    extra_dup = profile.get("duplicate_text_rows", 0) - profile["duplicate_rows"]
    if extra_dup > 0:
        add(Issue("dup_text", "Uniqueness", "Duplicate text (after normalisation)", severity_from_pct(_pct(extra_dup, n), 10, 2, 0.5),
                  extra_dup, _pct(extra_dup, n),
                  "Same text up to case/punctuation but differing in other fields.",
                  "Deduplicate on the normalised text; if labels differ see the label-conflict issue.",
                  "key = df[TEXT_COLS].fillna('').apply(' '.join, axis=1).str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)\n"
                  "df = df[~key.duplicated()]", step=4))
    nd = profile.get("near_duplicates", {})
    if nd.get("pairs"):
        k, checked = len(nd["pairs"]), nd["checked_rows"]
        add(Issue("near_dup", "Uniqueness", "Near-duplicate texts (TF-IDF cosine >= 0.9)", severity_from_pct(_pct(k, checked), 10, 2, 0.5),
                  k, _pct(k, checked),
                  f"{k} near-duplicate pairs among {checked} sampled rows (rewrites or minor edits of the same story).",
                  "Remove near duplicates or keep them in the same split (group split) so validation scores are not inflated.",
                  "# group near-duplicates and split by group\nfrom sklearn.model_selection import GroupShuffleSplit",
                  step=5, rows=[p[0] for p in nd["pairs"]][:500]))

    # 3. validity of text ---------------------------------------------------
    max_len = int(np.ceil(profile.get("combined_word_p95", 0))) or 64
    for name in text_cols:
        c = profile["columns"][name]
        art = c.get("artifacts", {})
        labels = {"broken_html_entity": "broken HTML entities (e.g. '#39;', '&amp;')", "html_tag": "HTML tags",
                  "url": "URLs", "backslash": "stray backslashes", "non_ascii": "non-ASCII characters",
                  "repeated_punct": "repeated punctuation",
                  "spreadsheet_error": "spreadsheet error codes (e.g. #NAME?)"}
        found = {k: v for k, v in art.items() if v}
        if found:
            total = max(found.values())
            add(Issue(f"artifacts_{name}", "Validity", f"Encoding / markup artifacts in '{name}'",
                      severity_from_pct(_pct(total, n), 40, 10, 1), total, _pct(total, n),
                      "Found " + ", ".join(f"{v} rows with {labels[k]}" for k, v in found.items()) + ".",
                      "Repair entities with html.unescape (after fixing bare '#39;' style entities), strip tags/URLs, "
                      "replace backslashes with spaces and normalise unicode (NFKC).",
                      "from dq.text_utils import clean_text\nfor col in TEXT_COLS:\n    df[col] = df[col].map(clean_text)", step=3))
        if c.get("shouting"):
            k = c["shouting"]
            add(Issue(f"shouting_{name}", "Validity", f"ALL-CAPS text in '{name}'", "low", k, _pct(k, n),
                      "Mostly upper-case rows look like headlines/boilerplate and create duplicate vocabulary entries.",
                      "Lower-case during tokenisation (already done by the recommended tokenizer).", "", step=6))
        hi = c.get("length_outliers_high", 0)
        if hi:
            add(Issue(f"len_outlier_{name}", "Validity", f"Unusually long texts in '{name}' (IQR rule)",
                      severity_from_pct(_pct(hi, n), 20, 8, 3), hi, _pct(hi, n),
                      f"95th percentile length is {c['char_len_p95']:.0f} chars / {c['word_len_p95']:.0f} words.",
                      f"Do not drop them: truncate the combined text at about {max_len} tokens (95th percentile) "
                      "and inspect the longest rows for concatenated articles.",
                      f"MAX_LEN = {max_len}", step=8))
    if profile.get("very_short_rows"):
        k = len(profile["very_short_rows"])
        add(Issue("very_short", "Validity", "Very short / truncated texts (< 3 words)", severity_from_pct(_pct(k, n), 10, 2, 0.5),
                  k, _pct(k, n), "Too little content to classify; often the result of truncation or scraping errors.",
                  "Drop or send for review.", "df = df[text.str.split().str.len() >= 3]", step=6,
                  rows=profile["very_short_rows"][:500]))

    # 4. sparsity -----------------------------------------------------------
    ts = profile.get("text_sparsity") or {}
    if ts:
        add(Issue("bow_sparsity", "Sparsity", "Bag-of-words representation is extremely sparse",
                  "info" if ts["sparsity_pct"] < 99 else "medium", ts["rows"], ts["sparsity_pct"],
                  f"{ts['sparsity_pct']:.3f}% of the {ts['rows']} x {ts['vocab_size']} document-term matrix is zero "
                  f"(about {ts['avg_unique_tokens_per_row']:.1f} distinct tokens per row). {ts['hapax_pct']:.1f}% of the "
                  f"vocabulary appears only once; {ts['vocab_for_95pct_coverage']} words cover 95% of all tokens.",
                  f"Keep sparse matrices (scipy CSR) for classic models, cap the vocabulary at about "
                  f"{ts['vocab_for_95pct_coverage']} words with min_freq >= 2, and prefer dense learned embeddings for deep models.",
                  f"TfidfVectorizer(min_df=2, max_features={ts['vocab_for_95pct_coverage']}, sublinear_tf=True)", step=7))
        if ts.get("high_oov_rows"):
            k = len(ts["high_oov_rows"])
            add(Issue("high_oov", "Validity", "Rows dominated by out-of-vocabulary tokens", severity_from_pct(_pct(k, n), 5, 1, 0.2),
                      k, _pct(k, n), f"Over 30% of the tokens are unknown to the reference vocabulary (mean OOV {ts['mean_oov_pct']:.1f}%). "
                      "Typical of gibberish, heavy typos, other languages or code.",
                      "Inspect and drop garbled rows; if the text is valid but domain-specific, extend the vocabulary.", "",
                      step=6, rows=ts["high_oov_rows"][:500]))
    if profile.get("numeric_zero_sparsity_pct", 0) > 50:
        z = profile["numeric_zero_sparsity_pct"]
        add(Issue("numeric_sparsity", "Sparsity", "Numeric features are mostly zero/missing", "medium", n, z,
                  f"{z:.1f}% of numeric cells are zero or missing.",
                  "Store as sparse, drop near-empty columns, or use models that handle sparsity (tree ensembles, linear with L1).",
                  "", step=7))

    # 5. numeric columns ----------------------------------------------------
    scales = []
    for name, c in profile["columns"].items():
        if c["kind"] == "numeric" and "std" in c:
            scales.append(c["std"])
            if c.get("outliers_iqr"):
                k = c["outliers_iqr"]
                add(Issue(f"outliers_{name}", "Validity", f"Outliers in '{name}' (IQR rule)",
                          severity_from_pct(_pct(k, n), 15, 5, 1), k, _pct(k, n),
                          f"Range {c['min']:.3g} to {c['max']:.3g}, skew {c['skew']:.2f}.",
                          "Check whether they are errors. Winsorise/clip at the 1st/99th percentile or use RobustScaler.",
                          f"lo, hi = df['{name}'].quantile([.01, .99])\ndf['{name}'] = df['{name}'].clip(lo, hi)", step=8))
            if abs(c.get("skew", 0)) > 2 and c.get("min", -1) >= 0:
                add(Issue(f"skew_{name}", "Distribution", f"Highly skewed '{name}'", "low", n, 100.0,
                          f"Skewness {c['skew']:.2f}.", "Apply log1p or Box-Cox before scaling.",
                          f"df['{name}'] = np.log1p(df['{name}'])", step=9))
        if c.get("is_constant"):
            add(Issue(f"constant_{name}", "Redundancy", f"Constant column '{name}'", "medium", n, 100.0,
                      "Only one distinct value, so it carries no information.", "Drop the column.",
                      f"df = df.drop(columns=['{name}'])", step=1))
        if c.get("looks_like_id"):
            add(Issue(f"id_{name}", "Redundancy", f"'{name}' looks like an identifier", "low", n, 100.0,
                      f"{c['unique_ratio']*100:.1f}% unique values.", "Exclude identifiers from model features.", "", step=1))
    if len(scales) >= 2 and max(scales) / max(min(scales), 1e-9) > 100:
        add(Issue("scaling", "Distribution", "Numeric features on very different scales", "medium", n, 100.0,
                  f"Standard deviations range from {min(scales):.3g} to {max(scales):.3g}.",
                  "Normalise (StandardScaler / MinMaxScaler) before training neural networks or distance-based models.",
                  "from sklearn.preprocessing import StandardScaler", step=9))
    for a, b, r in profile.get("correlation", {}).get("high_pairs", []):
        if "__" in a and "__" in b and a.split("__")[0] == b.split("__")[0]:
            continue  # chars vs words of the same text column are expected to correlate
        add(Issue(f"corr_{a}_{b}", "Redundancy", f"'{a}' and '{b}' are highly correlated", "low", n, abs(r) * 100,
                  f"Pearson r = {r:.3f}.", "Keep one of the pair to reduce redundancy/multicollinearity.", "", step=9))

    # 6. labels -------------------------------------------------------------
    lab = profile.get("label")
    if lab:
        if lab.get("invalid"):
            k = lab["invalid"]
            add(Issue("invalid_labels", "Label quality", "Labels outside the allowed set", severity_from_pct(_pct(k, n), 5, 1, 0.1),
                      k, _pct(k, n), f"Unexpected values: {', '.join(map(str, lab['invalid_values']))}.",
                      "Map them to a valid class if the mapping is obvious, otherwise drop the rows.",
                      f"df = df[df['{label_col}'].isin(VALID_LABELS)]", step=2, rows=lab["invalid_rows"][:500]))
        ir = lab.get("imbalance_ratio", 1)
        if ir > 1.5:
            add(Issue("imbalance", "Label quality", "Class imbalance", "high" if ir > 5 else "medium" if ir > 2 else "low",
                      n, 100 * (1 - lab["norm_entropy"]),
                      f"Largest/smallest class ratio {ir:.2f}; normalised entropy {lab['norm_entropy']:.3f}. Counts: {lab['counts']}.",
                      "Use stratified splits, class weights in the loss, or re-sampling; report macro-F1 rather than accuracy.",
                      "weights = len(y) / (n_classes * np.bincount(y))\nloss = nn.CrossEntropyLoss(weight=torch.tensor(weights))",
                      step=10))
        conflicts = profile.get("conflicting_label_rows", [])
        if conflicts:
            k = len(conflicts)
            add(Issue("label_conflict", "Label quality", "Same text with conflicting labels", severity_from_pct(_pct(k, n), 2, 0.5, 0.1),
                      k, _pct(k, n), "Identical (normalised) texts appear with different labels.",
                      "Resolve by majority vote or manual review; drop the group if no majority exists.", "", step=4,
                      rows=conflicts[:500]))

    if label_noise is not None and len(label_noise):
        flagged = label_noise[label_noise["flagged"]].sort_values("margin", ascending=False)
        k = len(flagged)
        if k:
            add(Issue("label_noise", "Label quality", "Likely mislabelled rows (BiGRU classifier, confident learning)",
                      severity_from_pct(_pct(k, n), 10, 3, 0.5), k, _pct(k, n),
                      f"The classifier confidently prefers another class for {k} rows "
                      f"(per-class thresholds {label_noise.attrs.get('thresholds')}; "
                      f"{label_noise.attrs.get('skipped_short', 0)} rows with under 12 tokens were not judged).",
                      "Review the top-ranked rows first. Relabel to the suggested class when correct, drop when ambiguous, "
                      "or train with a noise-robust loss (label smoothing).", "", step=11,
                      rows=flagged["row"].tolist()[:500]))
    if anomalies is not None and len(anomalies):
        flagged = anomalies[anomalies["flagged"]].sort_values("anomaly_z", ascending=False)
        k = len(flagged)
        if k:
            add(Issue("anomalies", "Validity", "Anomalous / garbled content (character autoencoder)",
                      severity_from_pct(_pct(k, n), 10, 3, 1), k, _pct(k, n),
                      "Length-adjusted reconstruction error above the 99th percentile of clean AG News text: "
                      "garbled strings, heavy typos, markup or content unlike news.",
                      "Inspect the highest-error rows; drop the garbage ones and keep genuine but unusual news.", "",
                      step=6, rows=flagged["row"].tolist()[:500]))

    issues.sort(key=lambda i: (SEVERITY_ORDER[i.severity], -i.pct))
    return issues


# ---------------------------------------------------------------- scoring
def _rows(issues, ids_prefix):
    return sum(i.affected for i in issues if any(i.id.startswith(p) for p in ids_prefix))


def quality_scores(profile, issues):
    n = max(profile["n_rows"], 1)
    completeness = 100 - profile["missing_cells_pct"]
    uniqueness = 100 - min(100, _pct(max(profile.get("duplicate_text_rows", 0), profile["duplicate_rows"]), n)
                           + (next((i.pct for i in issues if i.id == "near_dup"), 0)))
    validity = 100 - min(100, _pct(profile.get("artifact_rows", 0), n) * 0.5
                         + _pct(_rows(issues, ["very_short", "anomalies", "high_oov", "empty_text"]), n)
                         + sum(i.pct for i in issues if i.id.startswith("outliers_")) * 0.5)
    scores = {"Completeness": completeness, "Uniqueness": uniqueness, "Validity": validity}
    weights = {"Completeness": 0.3, "Uniqueness": 0.2, "Validity": 0.3}
    if profile.get("label"):
        bad = _rows(issues, ["invalid_labels", "label_conflict", "label_noise"]) + profile["label"]["missing"]
        scores["Label quality"] = 100 - min(100, _pct(bad, n) * 2)
        scores["Balance"] = 100 * profile["label"].get("norm_entropy", 1)
        weights.update({"Label quality": 0.25, "Balance": 0.1})
    total = sum(weights.values())
    overall = sum(scores[k] * weights[k] for k in scores) / total
    scores = {k: float(np.clip(v, 0, 100)) for k, v in scores.items()}
    return float(np.clip(overall, 0, 100)), scores


def grade(score):
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"


def pipeline(issues):
    steps, seen = [], set()
    for i in sorted(issues, key=lambda i: (i.step, SEVERITY_ORDER[i.severity])):
        if i.severity == "info" and not i.code:
            continue
        key = (i.step, i.recommendation)
        if key in seen:
            continue
        seen.add(key)
        steps.append(i)
    return steps


# ---------------------------------------------------------------- report
def build_report(profile, label_noise=None, anomalies=None, dataset_name="dataset"):
    issues = build_issues(profile, label_noise, anomalies)
    overall, scores = quality_scores(profile, issues)
    return {
        "dataset": dataset_name,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "overall": overall, "grade": grade(overall), "scores": scores,
        "issues": issues, "pipeline": pipeline(issues), "profile": profile,
    }


def to_markdown(report):
    p = report["profile"]
    out = [f"# Data Quality Report: {report['dataset']}",
           f"Generated {report['generated']}  ",
           f"**Overall score: {report['overall']:.1f}/100 (grade {report['grade']})**", "",
           f"Rows: {p['n_rows']:,}  Columns: {p['n_cols']}  Text columns: {', '.join(p['text_cols']) or 'none'}  "
           f"Label column: {p.get('label_col') or 'none'}", "", "## Quality dimensions", "",
           "| Dimension | Score |", "|---|---|"]
    out += [f"| {k} | {v:.1f} |" for k, v in report["scores"].items()]
    out += ["", "## Issues", "", "| Severity | Category | Issue | Affected | % |", "|---|---|---|---|---|"]
    out += [f"| {i.severity} | {i.category} | {i.title} | {i.affected:,} | {i.pct:.2f} |" for i in report["issues"]]
    out += ["", "## Recommended preprocessing pipeline", ""]
    for k, i in enumerate(report["pipeline"], 1):
        out.append(f"{k}. **{i.title}**: {i.recommendation}")
        if i.code:
            out += ["", "```python", i.code, "```", ""]
    out += ["", "## Issue details", ""]
    for i in report["issues"]:
        out += [f"### [{i.severity.upper()}] {i.title}", i.detail, "", f"Recommendation: {i.recommendation}", ""]
    return "\n".join(out)


def to_html(report):
    e = htmllib.escape
    p = report["profile"]
    dims = "".join(
        f"<div class='dim'><div class='dname'>{e(k)}</div><div class='bar'><span style='width:{v:.1f}%'></span></div>"
        f"<div class='dval'>{v:.1f}</div></div>" for k, v in report["scores"].items())
    rows = "".join(
        f"<tr><td><span class='sev' style='background:{SEVERITY_COLOR[i.severity]}'>{i.severity}</span></td>"
        f"<td>{e(i.category)}</td><td><b>{e(i.title)}</b><div class='muted'>{e(i.detail)}</div></td>"
        f"<td class='num'>{i.affected:,}</td><td class='num'>{i.pct:.2f}%</td></tr>" for i in report["issues"])
    steps = "".join(
        f"<li><b>{e(i.title)}</b><p>{e(i.recommendation)}</p>"
        + (f"<pre>{e(i.code)}</pre>" if i.code else "") + "</li>" for i in report["pipeline"])
    cols = "".join(
        f"<tr><td>{e(c['name'])}</td><td>{e(c['kind'])}</td><td>{e(c['dtype'])}</td>"
        f"<td class='num'>{c['missing_total']:,}</td><td class='num'>{c['missing_pct']:.2f}%</td>"
        f"<td class='num'>{c['n_unique']:,}</td></tr>" for c in p["columns"].values())
    ts = p.get("text_sparsity") or {}
    sparsity = (f"<p>Document-term matrix: {ts['rows']:,} rows x {ts['vocab_size']:,} terms, "
                f"<b>{ts['sparsity_pct']:.3f}% sparse</b>. {ts['vocab_for_95pct_coverage']:,} words cover 95% of tokens; "
                f"{ts['hapax_pct']:.1f}% of words occur once.</p>") if ts else ""
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Data Quality Report</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#f6f7f9;color:#101828}}
main{{max-width:1000px;margin:auto;padding:24px 16px}}
.card{{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:18px 20px;margin:14px 0}}
h1{{margin:0 0 4px}} h2{{margin:0 0 12px;font-size:18px}}
.score{{font-size:44px;font-weight:700}} .grade{{font-size:18px;color:#475467}}
.dim{{display:flex;align-items:center;gap:10px;margin:6px 0}} .dname{{width:130px}}
.bar{{flex:1;background:#eaecf0;border-radius:6px;height:10px;overflow:hidden}}
.bar span{{display:block;height:100%;background:#2e7d6b}} .dval{{width:50px;text-align:right}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{border-bottom:1px solid #eaecf0;padding:8px;text-align:left;vertical-align:top}}
.num{{text-align:right;white-space:nowrap}} .muted{{color:#667085;font-size:13px;margin-top:3px}}
.sev{{color:#fff;border-radius:4px;padding:2px 7px;font-size:12px;text-transform:uppercase}}
pre{{background:#f2f4f7;padding:10px;border-radius:6px;overflow-x:auto;font-size:12.5px}}
li p{{margin:4px 0}}
</style></head><body><main>
<div class='card'><h1>Data Quality Report</h1><div class='muted'>{e(report['dataset'])} | generated {report['generated']}</div>
<div style='display:flex;gap:40px;align-items:center;margin-top:12px;flex-wrap:wrap'>
<div><div class='score'>{report['overall']:.1f}</div><div class='grade'>out of 100, grade {report['grade']}</div></div>
<div style='flex:1;min-width:260px'>{dims}</div></div>
<p class='muted'>{p['n_rows']:,} rows, {p['n_cols']} columns. Text: {e(', '.join(p['text_cols']) or 'none')}. Label: {e(str(p.get('label_col')))}.</p></div>
<div class='card'><h2>Issues found ({len(report['issues'])})</h2><table><tr><th>Severity</th><th>Category</th><th>Issue</th><th>Affected</th><th>%</th></tr>{rows}</table></div>
<div class='card'><h2>Recommended preprocessing pipeline</h2><ol>{steps}</ol></div>
<div class='card'><h2>Sparsity</h2>{sparsity}<p>Missing cells: {p['missing_cells']:,} ({p['missing_cells_pct']:.2f}%).</p></div>
<div class='card'><h2>Column profile</h2><table><tr><th>Column</th><th>Kind</th><th>dtype</th><th>Missing</th><th>Missing %</th><th>Unique</th></tr>{cols}</table></div>
</main></body></html>"""


def issues_frame(issues):
    import pandas as pd
    return pd.DataFrame([{k: v for k, v in asdict(i).items() if k != "rows"} for i in issues])
