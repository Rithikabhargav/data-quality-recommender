"""Text cleaning, tokenisation, vocabulary and character n-gram features.

Everything the models need at inference time lives here so the notebook and the
Streamlit app use exactly the same preprocessing.
"""
import html
import json
import re
import unicodedata
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# AG News was scraped with broken HTML entities: "#39;" instead of "&#39;",
# "quot;" instead of "&quot;", and literal backslashes where line breaks were.
BARE_NUMERIC_ENTITY = re.compile(r"(?<!&)#(\d{2,4});")
BARE_NAMED_ENTITY = re.compile(r"(?<![&\w])(quot|amp|lt|gt|nbsp);", re.I)
HTML_TAG = re.compile(r"<[^>]{1,200}>")
URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
MULTI_SPACE = re.compile(r"\s+")
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Patterns used by the profiler to count artifacts in the raw text.
ARTIFACT_PATTERNS = {
    "broken_html_entity": re.compile(r"(?<!&)#\d{2,4};|(?<![&\w])(?:quot|amp|lt|gt|nbsp);|&(?:#\d+|[a-z]+);", re.I),
    "html_tag": re.compile(r"<[^>]{1,200}>|&lt;[^&]{1,200}&gt;", re.I),
    "url": URL,
    "backslash": re.compile(r"\\"),
    "non_ascii": re.compile(r"[^\x00-\x7f]"),
    "repeated_punct": re.compile(r"([!?.,])\1{3,}"),
    "spreadsheet_error": re.compile(r"#(?:NAME\?|VALUE!|REF!|DIV/0!|NUM!|NULL!)"),
}


def clean_text(text):
    """Repair encoding artifacts and normalise whitespace. Keeps case and punctuation."""
    if not isinstance(text, str):
        return ""
    s = BARE_NUMERIC_ENTITY.sub(r"&#\1;", text)
    s = BARE_NAMED_ENTITY.sub(r"&\1;", s)
    # unescape twice because some rows are double escaped (&amp;lt;b&amp;gt;)
    s = html.unescape(html.unescape(s))
    s = HTML_TAG.sub(" ", s)
    s = URL.sub(" ", s)
    s = s.replace("\\", " ")
    s = unicodedata.normalize("NFKC", s)
    return MULTI_SPACE.sub(" ", s).strip()


def tokenize(text):
    """Lower-case word tokens (letters, digits and simple contractions)."""
    return TOKEN.findall(text.lower()) if isinstance(text, str) else []


class Vocab:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, max_size=30000, min_freq=2):
        counts = Counter()
        for toks in token_lists:
            counts.update(toks)
        words = [w for w, c in counts.most_common(max_size - 2) if c >= min_freq]
        return cls([cls.PAD, cls.UNK] + words)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len):
        ids = [self.stoi.get(t, 1) for t in tokens[:max_len]]
        return ids + [0] * (max_len - len(ids))

    def encode_batch(self, texts, max_len):
        return np.array([self.encode(tokenize(t), max_len) for t in texts], dtype=np.int64)

    def oov_rate(self, tokens):
        if not tokens:
            return 1.0
        return sum(t not in self.stoi for t in tokens) / len(tokens)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


# Character 2-4 grams hashed into a fixed number of buckets. HashingVectorizer is
# stateless, so there is nothing to fit or pickle; the app rebuilds it identically.
CHAR_FEATURES = 4096


def char_ngram_matrix(texts):
    hv = HashingVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), n_features=CHAR_FEATURES,
        alternate_sign=False, norm="l2", lowercase=True,
    )
    return hv.transform(["" if not isinstance(t, str) else t for t in texts]).astype(np.float32)
