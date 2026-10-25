# Data Quality Report: AG News test set with injected issues
Generated 2026-09-29 19:43  
**Overall score: 86.1/100 (grade B)**

Rows: 7,866  Columns: 3  Text columns: Title, Description  Label column: Class Index

## Quality dimensions

| Dimension | Score |
|---|---|
| Completeness | 97.9 |
| Uniqueness | 96.2 |
| Validity | 69.8 |
| Label quality | 77.7 |
| Balance | 100.0 |

## Issues

| Severity | Category | Issue | Affected | % |
|---|---|---|---|---|
| high | Validity | Encoding / markup artifacts in 'Description' | 2,116 | 26.90 |
| high | Label quality | Likely mislabelled rows (BiGRU classifier, confident learning) | 762 | 9.69 |
| high | Validity | Anomalous / garbled content (character autoencoder) | 381 | 4.84 |
| high | Validity | Rows dominated by out-of-vocabulary tokens | 360 | 4.58 |
| high | Label quality | Same text with conflicting labels | 77 | 0.98 |
| medium | Sparsity | Bag-of-words representation is extremely sparse | 7,866 | 99.89 |
| medium | Validity | Encoding / markup artifacts in 'Title' | 757 | 9.62 |
| medium | Validity | Unusually long texts in 'Description' (IQR rule) | 299 | 3.80 |
| medium | Completeness | Missing values in 'Title' | 266 | 3.38 |
| medium | Validity | Unusually long texts in 'Title' (IQR rule) | 241 | 3.06 |
| medium | Completeness | Missing values in 'Description' | 228 | 2.90 |
| medium | Uniqueness | Exact duplicate rows | 152 | 1.93 |
| medium | Validity | Very short / truncated texts (< 3 words) | 118 | 1.50 |
| medium | Uniqueness | Near-duplicate texts (TF-IDF cosine >= 0.9) | 104 | 1.35 |
| medium | Completeness | Disguised missing values in 'Description' | 76 | 0.97 |
| medium | Completeness | Disguised missing values in 'Title' | 38 | 0.48 |
| medium | Label quality | Labels outside the allowed set | 30 | 0.38 |
| low | Validity | ALL-CAPS text in 'Title' | 58 | 0.74 |
| low | Uniqueness | Duplicate text (after normalisation) | 39 | 0.50 |
| low | Completeness | Missing values in 'Class Index' | 8 | 0.10 |
| low | Validity | ALL-CAPS text in 'Description' | 4 | 0.05 |

## Recommended preprocessing pipeline

1. **Disguised missing values in 'Description'**: Convert blanks and placeholder tokens to real NaN before any other step.

```python
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.replace({'n/a': np.nan, 'null': np.nan, 'none': np.nan, '?': np.nan})
```

2. **Missing values in 'Title'**: Text cannot be imputed meaningfully. Fill missing 'Title' with an empty string and rely on 'Description' for those rows; drop rows where every text field is empty.

```python
df['Title'] = df['Title'].replace(r'^\s*$', np.nan, regex=True).fillna('')
```

3. **Missing values in 'Description'**: Text cannot be imputed meaningfully. Fill missing 'Description' with an empty string and rely on 'Title' for those rows; drop rows where every text field is empty.

```python
df['Description'] = df['Description'].replace(r'^\s*$', np.nan, regex=True).fillna('')
```

4. **Missing values in 'Class Index'**: Rows without a label cannot be used for supervised training: drop them or send them for annotation.

```python
df = df.dropna(subset=['Class Index'])
```

5. **Labels outside the allowed set**: Map them to a valid class if the mapping is obvious, otherwise drop the rows.

```python
df = df[df['Class Index'].isin(VALID_LABELS)]
```

6. **Encoding / markup artifacts in 'Description'**: Repair entities with html.unescape (after fixing bare '#39;' style entities), strip tags/URLs, replace backslashes with spaces and normalise unicode (NFKC).

```python
from dq.text_utils import clean_text
for col in TEXT_COLS:
    df[col] = df[col].map(clean_text)
```

7. **Same text with conflicting labels**: Resolve by majority vote or manual review; drop the group if no majority exists.
8. **Exact duplicate rows**: Drop exact duplicates, keeping the first occurrence.

```python
df = df.drop_duplicates()
```

9. **Duplicate text (after normalisation)**: Deduplicate on the normalised text; if labels differ see the label-conflict issue.

```python
key = df[TEXT_COLS].fillna('').apply(' '.join, axis=1).str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)
df = df[~key.duplicated()]
```

10. **Near-duplicate texts (TF-IDF cosine >= 0.9)**: Remove near duplicates or keep them in the same split (group split) so validation scores are not inflated.

```python
# group near-duplicates and split by group
from sklearn.model_selection import GroupShuffleSplit
```

11. **Anomalous / garbled content (character autoencoder)**: Inspect the highest-error rows; drop the garbage ones and keep genuine but unusual news.
12. **Rows dominated by out-of-vocabulary tokens**: Inspect and drop garbled rows; if the text is valid but domain-specific, extend the vocabulary.
13. **Very short / truncated texts (< 3 words)**: Drop or send for review.

```python
df = df[text.str.split().str.len() >= 3]
```

14. **ALL-CAPS text in 'Title'**: Lower-case during tokenisation (already done by the recommended tokenizer).
15. **Bag-of-words representation is extremely sparse**: Keep sparse matrices (scipy CSR) for classic models, cap the vocabulary at about 14524 words with min_freq >= 2, and prefer dense learned embeddings for deep models.

```python
TfidfVectorizer(min_df=2, max_features=14524, sublinear_tf=True)
```

16. **Unusually long texts in 'Description' (IQR rule)**: Do not drop them: truncate the combined text at about 52 tokens (95th percentile) and inspect the longest rows for concatenated articles.

```python
MAX_LEN = 52
```

17. **Likely mislabelled rows (BiGRU classifier, confident learning)**: Review the top-ranked rows first. Relabel to the suggested class when correct, drop when ambiguous, or train with a noise-robust loss (label smoothing).

## Issue details

### [HIGH] Encoding / markup artifacts in 'Description'
Found 2116 rows with broken HTML entities (e.g. '#39;', '&amp;'), 293 rows with HTML tags, 243 rows with URLs, 739 rows with stray backslashes, 1 rows with repeated punctuation.

Recommendation: Repair entities with html.unescape (after fixing bare '#39;' style entities), strip tags/URLs, replace backslashes with spaces and normalise unicode (NFKC).

### [HIGH] Likely mislabelled rows (BiGRU classifier, confident learning)
The classifier confidently prefers another class for 762 rows (per-class thresholds {'World': 0.781, 'Sports': 0.814, 'Business': 0.745, 'Sci/Tech': 0.745}; 334 rows with under 12 tokens were not judged).

Recommendation: Review the top-ranked rows first. Relabel to the suggested class when correct, drop when ambiguous, or train with a noise-robust loss (label smoothing).

### [HIGH] Anomalous / garbled content (character autoencoder)
Length-adjusted reconstruction error above the 99th percentile of clean AG News text: garbled strings, heavy typos, markup or content unlike news.

Recommendation: Inspect the highest-error rows; drop the garbage ones and keep genuine but unusual news.

### [HIGH] Rows dominated by out-of-vocabulary tokens
Over 30% of the tokens are unknown to the reference vocabulary (mean OOV 5.5%). Typical of gibberish, heavy typos, other languages or code.

Recommendation: Inspect and drop garbled rows; if the text is valid but domain-specific, extend the vocabulary.

### [HIGH] Same text with conflicting labels
Identical (normalised) texts appear with different labels.

Recommendation: Resolve by majority vote or manual review; drop the group if no majority exists.

### [MEDIUM] Bag-of-words representation is extremely sparse
99.893% of the 7866 x 29280 document-term matrix is zero (about 31.4 distinct tokens per row). 54.3% of the vocabulary appears only once; 14524 words cover 95% of all tokens.

Recommendation: Keep sparse matrices (scipy CSR) for classic models, cap the vocabulary at about 14524 words with min_freq >= 2, and prefer dense learned embeddings for deep models.

### [MEDIUM] Encoding / markup artifacts in 'Title'
Found 757 rows with broken HTML entities (e.g. '#39;', '&amp;'), 154 rows with HTML tags, 133 rows with stray backslashes, 3 rows with spreadsheet error codes (e.g. #NAME?).

Recommendation: Repair entities with html.unescape (after fixing bare '#39;' style entities), strip tags/URLs, replace backslashes with spaces and normalise unicode (NFKC).

### [MEDIUM] Unusually long texts in 'Description' (IQR rule)
95th percentile length is 301 chars / 45 words.

Recommendation: Do not drop them: truncate the combined text at about 52 tokens (95th percentile) and inspect the longest rows for concatenated articles.

### [MEDIUM] Missing values in 'Title'
228 NaN; 38 placeholder tokens (e.g. 'n/a', 'null').

Recommendation: Text cannot be imputed meaningfully. Fill missing 'Title' with an empty string and rely on 'Description' for those rows; drop rows where every text field is empty.

### [MEDIUM] Unusually long texts in 'Title' (IQR rule)
95th percentile length is 65 chars / 11 words.

Recommendation: Do not drop them: truncate the combined text at about 52 tokens (95th percentile) and inspect the longest rows for concatenated articles.

### [MEDIUM] Missing values in 'Description'
152 NaN; 76 empty/whitespace strings.

Recommendation: Text cannot be imputed meaningfully. Fill missing 'Description' with an empty string and rely on 'Title' for those rows; drop rows where every text field is empty.

### [MEDIUM] Exact duplicate rows
Identical rows inflate the weight of some examples and leak between train and test splits.

Recommendation: Drop exact duplicates, keeping the first occurrence.

### [MEDIUM] Very short / truncated texts (< 3 words)
Too little content to classify; often the result of truncation or scraping errors.

Recommendation: Drop or send for review.

### [MEDIUM] Near-duplicate texts (TF-IDF cosine >= 0.9)
104 near-duplicate pairs among 7676 sampled rows (rewrites or minor edits of the same story).

Recommendation: Remove near duplicates or keep them in the same split (group split) so validation scores are not inflated.

### [MEDIUM] Disguised missing values in 'Description'
Empty strings or placeholder tokens are not counted by df.isna(), so they slip past naive checks.

Recommendation: Convert blanks and placeholder tokens to real NaN before any other step.

### [MEDIUM] Disguised missing values in 'Title'
Empty strings or placeholder tokens are not counted by df.isna(), so they slip past naive checks.

Recommendation: Convert blanks and placeholder tokens to real NaN before any other step.

### [MEDIUM] Labels outside the allowed set
Unexpected values: 0, 5, unknown.

Recommendation: Map them to a valid class if the mapping is obvious, otherwise drop the rows.

### [LOW] ALL-CAPS text in 'Title'
Mostly upper-case rows look like headlines/boilerplate and create duplicate vocabulary entries.

Recommendation: Lower-case during tokenisation (already done by the recommended tokenizer).

### [LOW] Duplicate text (after normalisation)
Same text up to case/punctuation but differing in other fields.

Recommendation: Deduplicate on the normalised text; if labels differ see the label-conflict issue.

### [LOW] Missing values in 'Class Index'
8 NaN.

Recommendation: Rows without a label cannot be used for supervised training: drop them or send them for annotation.

### [LOW] ALL-CAPS text in 'Description'
Mostly upper-case rows look like headlines/boilerplate and create duplicate vocabulary entries.

Recommendation: Lower-case during tokenisation (already done by the recommended tokenizer).
