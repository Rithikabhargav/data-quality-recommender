"""DQ-Recommender: Streamlit application.

Audit a CSV (or a bundled sample) with the statistical profiler and two PyTorch models, get a
scored data quality report, then let the preprocessing agent apply the fixes step by step and
measure how much they help a downstream model.

Run locally:  streamlit run app.py
"""
import html
import io
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dq.agent import (MISLABEL_POLICIES, PreprocessingAgent, agent_markdown, downstream_impact, stepwise_impact,
                      steps_frame)
from dq.dl_checks import DLChecker
from dq.profiler import combined_text, detect_label_column, detect_text_columns, profile_dataset
from dq.recommender import SEVERITY_COLOR, build_report, grade, issues_frame, to_html, to_markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
REFERENCE = os.path.join(HERE, "sample_data", "ag_news_reference.csv")
SAMPLES = {
    "AG News test set with injected issues (demo)": os.path.join(HERE, "sample_data", "ag_news_dirty_sample.csv"),
    "Original AG News test set": os.path.join(HERE, "sample_data", "ag_news_test.csv"),
}
ACTION_LABELS = {
    "standardise_missing": "Standardise disguised missing values", "drop_useless_columns": "Drop constant / ID columns",
    "drop_empty_text": "Drop rows with no text", "drop_invalid_labels": "Drop missing / invalid labels",
    "impute_text": "Fill missing text fields", "impute_other": "Impute numeric / categorical values",
    "repair_text": "Repair encoding & markup", "resolve_conflicts": "Resolve conflicting labels",
    "drop_duplicates": "Remove duplicates", "drop_near_duplicates": "Remove near duplicates",
    "drop_junk": "Remove garbled / off-domain rows", "drop_truncated": "Remove truncated texts",
    "fix_label_noise": "Fix suspected mislabels", "clip_outliers": "Clip numeric outliers",
    "log_transform": "Log-transform skewed columns",
}
# which agent action handles which issue (by issue id prefix)
FIXABLE = {"disguised_missing": "standardise_missing", "constant_": "drop_useless_columns", "id_": "drop_useless_columns",
           "empty_text": "drop_empty_text", "invalid_labels": "drop_invalid_labels", "missing_": "impute_text",
           "artifacts_": "repair_text", "label_conflict": "resolve_conflicts", "dup_": "drop_duplicates",
           "near_dup": "drop_near_duplicates", "anomalies": "drop_junk", "high_oov": "drop_junk",
           "very_short": "drop_truncated", "label_noise": "fix_label_noise", "outliers_": "clip_outliers",
           "skew_": "log_transform"}

NAV = [":material/dashboard: Overview", ":material/report: Issues", ":material/smart_toy: Preprocessing agent",
       ":material/table_chart: Data profile", ":material/psychology: Deep learning checks", ":material/download: Report & export"]


def go_to_agent():
    st.session_state["nav"] = NAV[2]


st.set_page_config(page_title="DQ-Recommender", page_icon=":material/fact_check:", layout="wide")
try:
    with open(os.path.join(ARTIFACTS, "config.json"), encoding="utf-8") as f:
        MODEL_INFO = json.load(f)
except OSError:
    MODEL_INFO = {}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
section[data-testid="stSidebar"] { background: #f4f7f6; border-right: 1px solid #e3e9e7; }
.hero { background: linear-gradient(120deg, #11453b 0%, #1f6f5f 55%, #2f8f7a 100%); color: #fff;
        border-radius: 16px; padding: 22px 28px; margin-bottom: 18px; }
.hero h1 { color: #fff; font-size: 1.9rem; margin: 0 0 4px 0; font-weight: 700; letter-spacing: -0.02em; }
.hero p { color: #d7ece6; margin: 0; font-size: 0.98rem; }
.hero .chips span { display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
        border-radius: 999px; padding: 3px 11px; margin: 10px 6px 0 0; font-size: .78rem; color: #fff; }
.card { background: #fff; border: 1px solid #e4e9ee; border-radius: 14px; padding: 16px 18px; height: 100%;
        box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.card h4 { margin: 0 0 6px 0; font-size: 1rem; }
.card p { margin: 0; color: #475467; font-size: .9rem; line-height: 1.45; }
.kpi { background: #fff; border: 1px solid #e4e9ee; border-radius: 14px; padding: 14px 16px; }
.kpi .label { color: #667085; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
.kpi .value { font-size: 1.65rem; font-weight: 700; color: #101828; line-height: 1.2; }
.kpi .sub { color: #667085; font-size: .8rem; }
.kpi .up { color: #1f7a4d; font-weight: 600; } .kpi .down { color: #b42318; font-weight: 600; }
.ring { width: 150px; height: 150px; border-radius: 50%; display: grid; place-items: center; position: relative;
        background: conic-gradient(var(--c) calc(var(--p) * 1%), #e8edf1 0); margin: auto; }
.ring::before { content: ""; position: absolute; inset: 13px; background: #fff; border-radius: 50%; }
.ring > div { position: relative; text-align: center; }
.ring b { font-size: 2.1rem; display: block; color: #101828; line-height: 1; }
.ring small { color: #667085; font-size: .8rem; }
.sev { color: #fff; border-radius: 6px; padding: 2px 8px; font-size: .7rem; text-transform: uppercase; font-weight: 600;
       letter-spacing: .03em; }
.fix { background: #e7f4ef; color: #1f6f5f; border: 1px solid #bfe0d4; border-radius: 6px; padding: 1px 7px;
       font-size: .72rem; font-weight: 600; margin-left: 6px; }
.issue { border: 1px solid #e4e9ee; border-radius: 12px; padding: 10px 14px; margin-bottom: 8px; background: #fff; }
.issue .t { font-weight: 600; } .issue .d { color: #475467; font-size: .88rem; margin-top: 3px; }
.loop { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 14px 0; }
.loop div { background: #f1f6f4; border: 1px solid #d7e6e0; border-radius: 10px; padding: 8px 12px; font-size: .85rem; }
.loop b { color: #1f6f5f; }
.step { border: 1px solid #e4e9ee; border-left: 5px solid var(--sc); border-radius: 12px; padding: 12px 16px;
        margin-bottom: 10px; background: #fff; }
.step .head { display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
.step .title { font-weight: 600; font-size: 1rem; }
.step .status { color: #fff; background: var(--sc); border-radius: 999px; padding: 1px 10px; font-size: .72rem;
        font-weight: 600; text-transform: uppercase; }
.step .row { margin-top: 6px; font-size: .88rem; color: #344054; }
.step .row span.k { color: #667085; font-weight: 600; margin-right: 4px; }
.step .why { margin-top: 8px; background: #f6faf8; border-radius: 8px; padding: 8px 10px; font-size: .86rem; color: #1d3b33; }
.step .metrics { margin-top: 8px; display: flex; gap: 14px; flex-wrap: wrap; font-size: .82rem; color: #475467; }
.step .metrics b { color: #101828; }
</style>""", unsafe_allow_html=True)

STATUS_COLOR = {"applied": "#1f7a4d", "rolled_back": "#b54708", "skipped": "#98a2b3", "advice": "#3b6ea8"}
GRADE_COLOR = {"A": "#1f7a4d", "B": "#4d8a2e", "C": "#b54708", "D": "#c4320a", "F": "#b42318"}


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading deep learning models...")
def load_checker():
    return DLChecker(ARTIFACTS, device="cpu")


@st.cache_data(show_spinner=False)
def read_csv(data, name):
    return pd.read_csv(io.BytesIO(data)) if data is not None else pd.read_csv(name)


@st.cache_data(show_spinner=False)
def load_reference():
    return pd.read_csv(REFERENCE) if os.path.exists(REFERENCE) else None


def esc(x):
    return html.escape(str(x))


def badge(sev):
    return f"<span class='sev' style='background:{SEVERITY_COLOR[sev]}'>{sev}</span>"


def fix_action(issue_id):
    return next((a for p, a in FIXABLE.items() if issue_id.startswith(p)), None)


def ring(score):
    g = grade(score)
    return (f"<div class='ring' style='--p:{score:.1f};--c:{GRADE_COLOR[g]}'><div><b>{score:.1f}</b>"
            f"<small>quality score<br>grade {g}</small></div></div>")


def kpi(label, value, sub=""):
    return f"<div class='kpi'><div class='label'>{esc(label)}</div><div class='value'>{value}</div><div class='sub'>{sub}</div></div>"


def delta(a, b, fmt="{:+.1f}", good_up=True):
    d = b - a
    cls = "up" if (d >= 0) == good_up else "down"
    return f"<span class='{cls}'>{fmt.format(d)}</span>"


def run_dl_checks(df, text_cols, label_col, max_rows, do_labels, do_anom):
    """Runs the DL checks on (a sample of) the rows; results are keyed by the DataFrame index."""
    checker = load_checker()
    idx = df.index.to_numpy()
    if len(df) > max_rows:
        idx = np.sort(np.random.default_rng(0).choice(idx, max_rows, replace=False))
    texts = combined_text(df.loc[idx], text_cols).reset_index(drop=True)
    noise = anomalies = None
    if do_labels and label_col:
        noise = checker.label_noise(texts, df.loc[idx, label_col].reset_index(drop=True))
        if noise is not None:
            attrs = dict(noise.attrs)
            noise["row"] = idx[noise["row"].to_numpy()]
            noise.attrs.update(attrs)
    if do_anom and text_cols:
        anomalies = checker.anomalies(texts)
        if anomalies is not None:
            anomalies["row"] = idx[anomalies["row"].to_numpy()]
    return noise, anomalies, len(idx)


def audit(df, text_cols, label_col, valid_labels, max_rows, do_labels, do_anom, name):
    vocab = load_checker().vocab if text_cols else None
    prof = profile_dataset(df, text_cols, label_col, valid_labels=valid_labels, vocab=vocab)
    noise = anomalies = None
    n_dl = 0
    if text_cols and (do_labels or do_anom):
        noise, anomalies, n_dl = run_dl_checks(df, text_cols, label_col, max_rows, do_labels, do_anom)
    return build_report(prof, noise, anomalies, dataset_name=name), noise, anomalies, n_dl


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### :material/database: Data")
    source = st.radio("Source", list(SAMPLES) + ["Upload a CSV"], index=0, label_visibility="collapsed")
    upload = st.file_uploader("CSV file", type=["csv"]) if source == "Upload a CSV" else None
    if source == "Upload a CSV" and upload is None:
        st.info("Upload a CSV to continue.")
        st.stop()
    raw = read_csv(upload.getvalue() if upload else None, upload.name if upload else SAMPLES[source])
    df = raw.reset_index(drop=True)
    name = upload.name if upload else source

    st.markdown("### :material/view_column: Columns")
    guess_text = detect_text_columns(df)
    text_cols = st.multiselect("Text columns", list(df.columns), default=guess_text)
    guess_label = detect_label_column(df, text_cols)
    options = ["(none)"] + [c for c in df.columns if c not in text_cols]
    label_col = st.selectbox("Label column", options, index=options.index(guess_label) if guess_label in options else 0)
    label_col = None if label_col == "(none)" else label_col
    default_valid = ""
    if label_col and {"1", "2", "3", "4"} <= set(df[label_col].dropna().astype(str)):
        default_valid = "1,2,3,4"
    valid_txt = st.text_input("Allowed label values (optional)", default_valid)
    valid_labels = [v.strip() for v in valid_txt.split(",") if v.strip()] or None

    st.markdown("### :material/psychology: Deep learning checks")
    do_labels = st.toggle("Label noise (BiGRU classifier)", value=label_col is not None, disabled=label_col is None,
                          help="Needs AG News style labels: 1-4, 0-3 or World/Sports/Business/Sci/Tech.")
    do_anom = st.toggle("Content anomalies (autoencoder)", value=bool(text_cols), disabled=not text_cols)
    max_rows = st.slider("Max rows for DL checks", 1000, 50000, 20000, 1000,
                         help="Larger files are sampled so the app stays responsive on a CPU.")
    run = st.button("Run data quality audit", type="primary", width="stretch", icon=":material/play_arrow:")

# ---------------------------------------------------------------- header
acc = MODEL_INFO.get("test_accuracy", 0)
st.markdown(f"""
<div class="hero"><h1>DQ-Recommender</h1>
<p>Find data quality problems in a text dataset, get a scored report, and let the preprocessing agent fix them.</p>
<div class="chips"><span>Statistical profiler</span><span>BiGRU label auditor ({100 * acc:.1f}% test acc.)</span>
<span>Character autoencoder</span><span>Preprocessing agent</span><span>PyTorch</span></div></div>""",
            unsafe_allow_html=True)

key = (name, tuple(text_cols), label_col, tuple(valid_labels or []), do_labels, do_anom, max_rows, len(df))
if run:
    with st.status("Auditing dataset...", expanded=True) as status:
        st.write("Profiling columns, duplicates, sparsity and artifacts")
        st.write("Running deep learning checks")
        report, noise, anomalies, n_dl = audit(df, text_cols, label_col, valid_labels, max_rows, do_labels, do_anom, name)
        if do_labels and noise is None:
            st.warning("Label values could not be mapped to the AG News classes, so the label-noise check was skipped.")
        st.write("Scoring issues and building recommendations")
        st.session_state["result"] = dict(key=key, report=report, noise=noise, anomalies=anomalies, n_dl=n_dl,
                                          valid_labels=valid_labels)
        st.session_state.pop("agent", None)
        st.session_state["nav"] = NAV[0]
        status.update(label="Audit complete", state="complete", expanded=False)

res = st.session_state.get("result")
if res is None or res["key"] != key:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (":material/query_stats:", "Statistical checks", "Missing and disguised-missing values, duplicates, conflicting "
         "labels, sparsity, vocabulary coverage, HTML/encoding artifacts, outliers, class balance, correlations."),
        (":material/label:", "Label auditor", f"A BiGRU topic classifier ({100 * acc:.1f}% test accuracy) flags rows whose "
         "label it confidently disagrees with (confident learning)."),
        (":material/blur_on:", "Anomaly detector", "A denoising autoencoder on character n-grams flags garbled or "
         "off-domain text; rows with over 30% unknown words are flagged too."),
        (":material/smart_toy:", "Preprocessing agent", "Applies the fixes one at a time, re-audits after each, keeps "
         "only changes that help, and measures the effect on a downstream model."),
    ]
    for col, (_, t, d) in zip((c1, c2, c3, c4), cards):
        col.markdown(f"<div class='card'><h4>{t}</h4><p>{d}</p></div>", unsafe_allow_html=True)
    st.markdown("#### How it works")
    st.graphviz_chart("""
digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#eef5f2", color="#1f6f5f", fontname="Helvetica", fontsize=11];
  csv [label="Your CSV"]; prof [label="Statistical profiler"]; clf [label="BiGRU label auditor"]; ae [label="Char autoencoder"];
  rec [label="Recommender\\nissues + score"]; agent [label="Preprocessing agent\\nobserve, plan, act, verify", fillcolor="#fff4e5", color="#b54708"];
  out [label="Clean data, report,\\nimpact on a model", fillcolor="#e8f0fb", color="#3b6ea8"];
  csv -> prof -> rec; csv -> clf -> rec; csv -> ae -> rec; rec -> agent -> out; agent -> rec [style=dashed, label=" re-audit", fontsize=9]; }""")
    st.info("Pick a data source and columns in the sidebar, then press **Run data quality audit**.", icon=":material/info:")
    st.markdown("#### Data preview")
    st.dataframe(df.head(50), width="stretch")
    st.stop()

report, noise, anomalies = res["report"], res["noise"], res["anomalies"]
prof, issues = report["profile"], report["issues"]
fixable = [i for i in issues if fix_action(i.id)]

# a segmented control keeps the selected section across reruns (st.tabs jumps back to the first tab)
st.session_state.setdefault("nav", NAV[0])
page = st.segmented_control("Section", NAV, key="nav", label_visibility="collapsed") or NAV[0]

# ---------------------------------------------------------------- overview
if page == NAV[0]:
    left, right = st.columns([1, 3])
    with left:
        st.markdown(ring(report["overall"]), unsafe_allow_html=True)
    with right:
        k = st.columns(4)
        k[0].markdown(kpi("Rows", f"{prof['n_rows']:,}", f"{prof['n_cols']} columns"), unsafe_allow_html=True)
        k[1].markdown(kpi("Issues found", len(issues), f"{sum(i.severity in ('critical', 'high') for i in issues)} critical/high"),
                      unsafe_allow_html=True)
        k[2].markdown(kpi("Auto-fixable", len(fixable), "by the preprocessing agent"), unsafe_allow_html=True)
        k[3].markdown(kpi("Missing cells", f"{prof['missing_cells_pct']:.2f}%", f"{prof['missing_cells']:,} cells"),
                      unsafe_allow_html=True)
        dims = pd.DataFrame({"dimension": list(report["scores"]), "score": list(report["scores"].values())})
        chart = alt.Chart(dims).mark_bar(cornerRadiusEnd=5, color="#1f6f5f", size=20).encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]), title=None),
            y=alt.Y("dimension:N", sort="-x", title=None), tooltip=["dimension", alt.Tooltip("score:Q", format=".1f")])
        text = chart.mark_text(align="left", dx=4, color="#344054").encode(text=alt.Text("score:Q", format=".1f"))
        st.altair_chart((chart + text).properties(height=48 * len(dims) + 20, title="Quality dimensions"), width="stretch")
    st.markdown("#### Top issues")
    for i in issues[:6]:
        fx = "<span class='fix'>auto-fix</span>" if fix_action(i.id) else ""
        st.markdown(f"<div class='issue'>{badge(i.severity)} <span class='t'>{esc(i.title)}</span>{fx}"
                    f"<span style='float:right;color:#667085'>{i.affected:,} rows | {i.pct:.2f}%</span>"
                    f"<div class='d'>{esc(i.recommendation)}</div></div>", unsafe_allow_html=True)
    if fixable:
        st.success(f"{len(fixable)} of these issues can be fixed automatically by the preprocessing agent.",
                   icon=":material/smart_toy:")
        st.button("Open the preprocessing agent", icon=":material/arrow_forward:", on_click=go_to_agent)
    if res["n_dl"] and res["n_dl"] < prof["n_rows"]:
        st.caption(f"Deep learning checks ran on a random sample of {res['n_dl']:,} rows.")

# ---------------------------------------------------------------- issues
if page == NAV[1]:
    cats = sorted({i.category for i in issues})
    pick = st.multiselect("Filter by category", cats, default=cats)
    for i in [i for i in issues if i.category in pick]:
        fx = f" | auto-fix: {ACTION_LABELS[fix_action(i.id)]}" if fix_action(i.id) else ""
        with st.expander(f"[{i.severity.upper()}] {i.title}  ({i.affected:,} | {i.pct:.2f}%){fx}"):
            st.markdown(f"{badge(i.severity)} &nbsp; *{i.category}*", unsafe_allow_html=True)
            st.write(i.detail)
            st.markdown(f"**Recommendation:** {i.recommendation}")
            if i.code:
                st.code(i.code, language="python")
            if i.rows:
                st.caption(f"Example rows ({min(len(i.rows), 25)} of {len(i.rows)})")
                st.dataframe(df.iloc[[r for r in i.rows[:25] if r < len(df)]], width="stretch")

# ---------------------------------------------------------------- agent
if page == NAV[2]:
    st.markdown("#### Preprocessing agent")
    st.markdown("""<div class="loop"><div><b>1. Observe</b> audit the current data</div><div><b>2. Plan</b> pick the next
    issue it can fix</div><div><b>3. Act</b> apply the fix to a copy</div><div><b>4. Verify</b> re-audit; keep it or roll
    back</div><div><b>5. Measure</b> train a model before vs after</div></div>""", unsafe_allow_html=True)
    with st.expander("Agent settings", icon=":material/tune:", expanded=False):
        s1, s2 = st.columns(2)
        budget = s1.slider("Maximum share of rows the agent may remove", 5, 40, 20, 1, format="%d%%") / 100
        policy = s2.selectbox("Suspected mislabels", list(MISLABEL_POLICIES), format_func=MISLABEL_POLICIES.get)
        conf = s2.slider("Relabel only when the classifier is at least this confident", 0.70, 0.99, 0.90, 0.01)
        allowed = s1.multiselect("Actions the agent may use", list(ACTION_LABELS), default=list(ACTION_LABELS),
                                 format_func=ACTION_LABELS.get)
        measure = st.toggle("Measure the effect on a downstream model (TF-IDF + logistic regression)", value=bool(label_col))
        stepwise = st.toggle("Also measure each step's contribution (slower: retrains the model after every step)",
                             value=False, disabled=not measure)
    go = st.button("Run preprocessing agent", type="primary", icon=":material/smart_toy:", disabled=not fixable)
    if not fixable:
        st.info("No automatically fixable issues were found.")

    if go:
        agent = PreprocessingAgent(prof["text_cols"], prof.get("label_col"), res["valid_labels"],
                                   load_checker().vocab if prof["text_cols"] else None, noise, anomalies,
                                   max_row_loss=budget, mislabel_policy=policy, relabel_confidence=conf,
                                   allowed=allowed, keep_snapshots=measure and stepwise)
        with st.status("Agent working...", expanded=True) as status:
            def show(step):
                icon = {"applied": "check_circle", "rolled_back": "undo", "skipped": "skip_next", "advice": "lightbulb"}[step.status]
                st.write(f":material/{icon}: **{step.title}**: {step.detail}")
            cleaned, steps, first, last = agent.run(df, on_step=show)
            st.write(":material/fact_check: Re-auditing the cleaned data with fresh deep learning checks")
            final_report, f_noise, f_anom, _ = audit(cleaned.reset_index(drop=True), prof["text_cols"], prof.get("label_col"),
                                                     res["valid_labels"], max_rows, do_labels, do_anom, f"{name} (after agent)")
            impact = None
            if measure and prof.get("label_col") and prof["text_cols"]:
                st.write(":material/model_training: Training a downstream model on the data before and after")
                ref = load_reference() if (noise is not None and load_checker().map_labels(df[prof["label_col"]]) is not None) else None
                try:
                    impact = downstream_impact(df, cleaned, prof["text_cols"], prof["label_col"], reference=ref,
                                               repaired="repair_text" in {s.action for s in steps if s.status == "applied"})
                    if stepwise:
                        st.write(":material/stacked_line_chart: Retraining after every applied step")
                        impact["stepwise"] = stepwise_impact(agent.snapshots, prof["text_cols"], prof["label_col"], ref)
                except ValueError as err:
                    st.warning(f"Could not measure downstream impact: {err}")
            status.update(label="Agent finished", state="complete", expanded=False)
        st.session_state["agent"] = dict(key=key, cleaned=cleaned.reset_index(drop=True), steps=steps, first=first,
                                         last=last, final=final_report, impact=impact)

    ag = st.session_state.get("agent")
    if ag and ag["key"] == key:
        steps, final = ag["steps"], ag["final"]
        applied = [s for s in steps if s.status == "applied"]
        n_before, n_after = prof["n_rows"], len(ag["cleaned"])
        k = st.columns(5)
        k[0].markdown(kpi("Quality score", f"{report['overall']:.1f} &rarr; {final['overall']:.1f}",
                          delta(report["overall"], final["overall"]) + f" | grade {report['grade']} &rarr; {final['grade']}"),
                      unsafe_allow_html=True)
        k[1].markdown(kpi("Rows", f"{n_before:,} &rarr; {n_after:,}",
                          f"{100 * (n_before - n_after) / max(n_before, 1):.1f}% removed"), unsafe_allow_html=True)
        k[2].markdown(kpi("Issues", f"{len(issues)} &rarr; {len(final['issues'])}",
                          delta(len(issues), len(final["issues"]), "{:+d}", good_up=False)), unsafe_allow_html=True)
        k[3].markdown(kpi("Steps applied", len(applied), f"{sum(s.cells_changed for s in applied):,} cells changed"),
                      unsafe_allow_html=True)
        imp = ag["impact"]
        if imp:
            k[4].markdown(kpi("Downstream accuracy", f"{100 * imp['before']['accuracy']:.1f}% &rarr; {100 * imp['after']['accuracy']:.1f}%",
                              delta(100 * imp["before"]["accuracy"], 100 * imp["after"]["accuracy"], "{:+.2f} pts")),
                          unsafe_allow_html=True)
        else:
            k[4].markdown(kpi("Downstream accuracy", "n/a", "not measured"), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        dims = pd.DataFrame([{"dimension": d, "when": w, "score": v}
                             for w, sc in (("before", report["scores"]), ("after", final["scores"])) for d, v in sc.items()])
        c1.altair_chart(alt.Chart(dims).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("dimension:N", title=None), x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]), title=None),
            yOffset="when:N", color=alt.Color("when:N", scale=alt.Scale(domain=["before", "after"], range=["#d0d5dd", "#1f6f5f"]),
                                              title=None, legend=alt.Legend(orient="bottom")),
            tooltip=["dimension", "when", alt.Tooltip("score:Q", format=".1f")]).properties(
            height=260, title="Quality dimensions before and after"), width="stretch")
        traj = pd.DataFrame([{"step": 0, "label": "start", "score": ag["first"].score, "rows": n_before}] +
                            [{"step": k + 1, "label": s.title, "score": s.score_after, "rows": s.rows_after}
                             for k, s in enumerate(applied)])
        base = alt.Chart(traj).encode(x=alt.X("step:O", title="applied step", axis=alt.Axis(labelAngle=0)))
        c2.altair_chart((base.mark_line(color="#1f6f5f", point=True).encode(
            y=alt.Y("score:Q", scale=alt.Scale(zero=False), title="score"), tooltip=["label", alt.Tooltip("score:Q", format=".1f"), "rows"])
        ).properties(height=260, title="Agent trajectory: score after each applied step"), width="stretch")
        st.caption("Step scores use the deep learning findings from the first audit; the final score comes from a fresh "
                   "re-audit of the cleaned data, so it can be slightly lower than the last step.")

        st.markdown("#### What the agent did and why it helps")
        for n_, s in enumerate(steps, 1):
            metrics = ""
            if s.status in ("applied", "rolled_back"):
                metrics = (f"<div class='metrics'><span>rows <b>{s.rows_before:,} &rarr; {s.rows_after:,}</b></span>"
                           f"<span>cells changed <b>{s.cells_changed:,}</b></span>"
                           f"<span>issue size <b>{s.target_before:,} &rarr; {s.target_after:,}</b></span>"
                           f"<span>score <b>{s.score_before:.1f} &rarr; {s.score_after:.1f}</b> ({s.score_after - s.score_before:+.1f})</span></div>")
            st.markdown(f"""<div class="step" style="--sc:{STATUS_COLOR[s.status]}">
<div class="head"><span class="title">{n_}. {esc(s.title)}</span><span class="status">{s.status.replace('_', ' ')}</span></div>
<div class="row"><span class="k">Observed</span>{esc(s.observation)}</div>
<div class="row"><span class="k">Action</span>{esc(s.detail)}</div>{metrics}
<div class="why"><b>Why it helps:</b> {esc(s.why)}</div></div>""", unsafe_allow_html=True)

        st.markdown("#### How the preprocessing helps the final output")
        if imp:
            b, a = imp["before"], imp["after"]
            m1, m2 = st.columns([1, 1])
            def bar_row(label, before_v, after_v):
                lo_ = max(0.0, min(before_v, after_v) - 0.05)
                w = lambda v: 100 * (v - lo_) / (1 - lo_)
                return (f"<div style='margin:10px 0 14px'><div style='font-weight:600;margin-bottom:4px'>{label}</div>"
                        f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'><span style='width:90px;color:#667085;"
                        f"font-size:.82rem'>before agent</span><div style='flex:1;background:#f2f4f7;border-radius:6px'>"
                        f"<div style='width:{w(before_v):.1f}%;background:#c7ccd4;height:18px;border-radius:6px'></div></div>"
                        f"<b style='width:64px;text-align:right'>{100 * before_v:.2f}%</b></div>"
                        f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'><span style='width:90px;color:#667085;"
                        f"font-size:.82rem'>after agent</span><div style='flex:1;background:#f2f4f7;border-radius:6px'>"
                        f"<div style='width:{w(after_v):.1f}%;background:#1f6f5f;height:18px;border-radius:6px'></div></div>"
                        f"<b style='width:64px;text-align:right'>{100 * after_v:.2f}%</b></div></div>")
            m1.markdown("<div class='card'><h4>Same model, trained before vs after the agent</h4>"
                        + bar_row("Accuracy", b["accuracy"], a["accuracy"])
                        + bar_row("Macro-F1", b["macro_f1"], a["macro_f1"])
                        + "<p style='font-size:.8rem'>Bars start just below the lower value so the difference is visible.</p></div>",
                        unsafe_allow_html=True)
            m2.markdown(f"""
The same quick model (TF-IDF + logistic regression) was trained on the data **before** and **after** the agent and
scored on the same evaluation data ({imp['method']}).

* Accuracy **{100 * b['accuracy']:.2f}% &rarr; {100 * a['accuracy']:.2f}%** ({100 * (a['accuracy'] - b['accuracy']):+.2f} points),
  macro-F1 **{b['macro_f1']:.3f} &rarr; {a['macro_f1']:.3f}**,
  while training on **{a['rows']:,}** instead of **{b['rows']:,}** rows.
* A better model from **fewer** rows means the removed and corrected rows were hurting it.
* On AG News most of this gain comes from **fixing mislabels**: wrong labels teach the model wrong answers.
  Turn on "measure each step's contribution" in the settings to see the effect of each step.
""")
        else:
            st.info("Enable 'Measure the effect on a downstream model' in the agent settings (needs a label column).")
        # dataset-level effects: many steps help in ways a single accuracy number does not show
        fp = final["profile"]
        ts0, ts1 = prof.get("text_sparsity") or {}, fp.get("text_sparsity") or {}
        issue_n = lambda rep_, pre: sum(i.affected for i in rep_["issues"] if i.id.startswith(pre))
        facts = [("Rows", prof["n_rows"], fp["n_rows"], "{:,}"),
                 ("Missing cells", prof["missing_cells"], fp["missing_cells"], "{:,}"),
                 ("Duplicate texts", prof.get("duplicate_text_rows", 0), fp.get("duplicate_text_rows", 0), "{:,}"),
                 ("Rows with encoding / markup artifacts", prof.get("artifact_rows", 0), fp.get("artifact_rows", 0), "{:,}"),
                 ("Vocabulary size", ts0.get("vocab_size", 0), ts1.get("vocab_size", 0), "{:,}"),
                 ("Words seen only once", ts0.get("hapax_pct", 0), ts1.get("hapax_pct", 0), "{:.1f}%"),
                 ("Suspected mislabels", issue_n(report, "label_noise"), issue_n(final, "label_noise"), "{:,}"),
                 ("Garbled / off-domain rows", issue_n(report, "anomalies"), issue_n(final, "anomalies"), "{:,}")]
        st.markdown("**What changed in the dataset**")
        st.dataframe(pd.DataFrame([{"measure": m, "before": f.format(x), "after": f.format(y)} for m, x, y, f in facts]),
                     hide_index=True, width="stretch")
        if imp and "stepwise" in imp:
            sw = imp["stepwise"].dropna(subset=["change"]).copy()
            sw["change_pts"] = 100 * sw["change"]
            st.altair_chart(alt.Chart(sw).mark_bar(cornerRadiusEnd=3).encode(
                y=alt.Y("after step:N", sort=None, title=None), x=alt.X("change_pts:Q", title="accuracy change (points)"),
                color=alt.condition("datum.change_pts >= 0", alt.value("#1f6f5f"), alt.value("#c4320a")),
                tooltip=["after step", "rows", alt.Tooltip("accuracy:Q", format=".4f"), alt.Tooltip("change_pts:Q", format="+.2f")]
            ).properties(height=34 * len(sw) + 30, title="Contribution of each applied step to downstream accuracy"), width="stretch")
            se = 100 * np.sqrt(0.88 * 0.12 / 5000)
            st.caption(f"Changes smaller than about {se:.1f} points (one standard error on 5,000 evaluation rows) are within noise.")
        st.markdown("""
**How each kind of step helps, and where you see it**

| Step type | What it fixes | Where the benefit shows |
|---|---|---|
| Standardise / impute missing values | NaN and fake 'N/A' tokens | Pipelines no longer crash or learn 'N/A' as a word; rows (and their labels) are kept |
| Drop invalid labels, resolve conflicts, fix mislabels | Wrong or contradictory targets | **Accuracy**: the loss rewards the right answer (the largest gain on AG News) |
| Remove duplicates and near duplicates | Leakage between train and test | **Trustworthy evaluation**: validation scores stop being inflated |
| Repair encodings and markup | Junk tokens such as '#39;' | **Smaller, cleaner vocabulary**; needed by pretrained tokenizers and for readable outputs |
| Remove junk and truncated rows | Gibberish, off-domain and near-empty rows | Less noise to fit and a cleaner vocabulary |
| Advice (vocabulary size, sequence length, class weights) | Representation choices | Faster training with almost no information lost |
""")
        remaining = [i for i in final["issues"] if i.severity in ("critical", "high", "medium")]
        if remaining:
            with st.expander(f"Issues that still need attention ({len(remaining)})", icon=":material/warning:"):
                for i in remaining:
                    st.markdown(f"{badge(i.severity)} **{esc(i.title)}**: {esc(i.recommendation)}", unsafe_allow_html=True)
        st.markdown("#### Downloads")
        d = st.columns(4)
        d[0].download_button("Cleaned CSV", ag["cleaned"].to_csv(index=False).encode(), "cleaned.csv", "text/csv",
                             icon=":material/table:", width="stretch")
        d[1].download_button("Agent log (Markdown)", agent_markdown(steps, ag["first"], ag["last"], imp), "agent_log.md",
                             "text/markdown", icon=":material/description:", width="stretch")
        d[2].download_button("Steps (CSV)", steps_frame(steps).to_csv(index=False), "agent_steps.csv", "text/csv",
                             icon=":material/list:", width="stretch")
        d[3].download_button("Report after cleaning (HTML)", to_html(final), "report_after_agent.html", "text/html",
                             icon=":material/web:", width="stretch")
        with st.expander("Preview cleaned data", icon=":material/visibility:"):
            st.dataframe(ag["cleaned"].head(200), width="stretch")

# ---------------------------------------------------------------- data profile
if page == NAV[3]:
    sub = st.segmented_control("View", ["Columns", "Missing values", "Text & sparsity", "Correlation"],
                               default="Columns") or "Columns"
    if sub == "Columns":
        cols = pd.DataFrame([{k: v for k, v in c.items() if not isinstance(v, dict)} for c in prof["columns"].values()])
        st.dataframe(cols, width="stretch", hide_index=True)
    elif sub == "Missing values":
        cols = pd.DataFrame([{k: v for k, v in c.items() if not isinstance(v, dict)} for c in prof["columns"].values()])
        miss = cols[["name", "missing_nan", "missing_blank", "missing_placeholder"]].melt("name", var_name="type", value_name="rows")
        if miss["rows"].sum():
            st.altair_chart(alt.Chart(miss).mark_bar().encode(
                x=alt.X("rows:Q", title="Rows"), y=alt.Y("name:N", title=None),
                color=alt.Color("type:N", title="Missing as", scale=alt.Scale(range=["#1f6f5f", "#e9a23b", "#b42318"]))).properties(
                title="Missing values by column"), width="stretch")
            head = df.head(300)
            sample = head.isna() | head.astype(str).apply(lambda s: s.str.strip().isin(["", "N/A", "n/a", "null", "None"]))
            grid = sample.reset_index().melt("index", var_name="column", value_name="missing")
            st.altair_chart(alt.Chart(grid).mark_rect().encode(
                x=alt.X("column:N", title=None), y=alt.Y("index:O", title="Row (first 300)", axis=None),
                color=alt.Color("missing:N", scale=alt.Scale(domain=[False, True], range=["#e8edf1", "#b42318"]))).properties(
                height=300, title="Missingness matrix"), width="stretch")
        else:
            st.success("No missing values detected (including blanks and placeholder tokens).")
    elif sub == "Text & sparsity":
        ts = prof.get("text_sparsity") or {}
        if not ts:
            st.info("No text columns selected.")
        else:
            k = st.columns(4)
            k[0].markdown(kpi("BoW sparsity", f"{ts['sparsity_pct']:.3f}%", "share of zeros"), unsafe_allow_html=True)
            k[1].markdown(kpi("Vocabulary", f"{ts['vocab_size']:,}", "distinct tokens"), unsafe_allow_html=True)
            k[2].markdown(kpi("Words for 95% coverage", f"{ts['vocab_for_95pct_coverage']:,}", "suggested vocab size"), unsafe_allow_html=True)
            k[3].markdown(kpi("Words seen once", f"{ts['hapax_pct']:.1f}%", "long tail"), unsafe_allow_html=True)
            a, b = st.columns(2)
            cov = pd.DataFrame(ts["coverage_curve"], columns=["vocab_size", "coverage"])
            a.altair_chart(alt.Chart(cov).mark_area(line={"color": "#1f6f5f"}, color="#cfe6de").encode(
                x=alt.X("vocab_size:Q", scale=alt.Scale(type="log"), title="Vocabulary size (log)"),
                y=alt.Y("coverage:Q", axis=alt.Axis(format="%"), title="Token coverage")).properties(title="Vocabulary coverage"),
                width="stretch")
            top = pd.DataFrame(list(ts["top_tokens"].items()), columns=["token", "count"])
            b.altair_chart(alt.Chart(top).mark_bar(color="#3b6ea8", cornerRadiusEnd=3).encode(
                x="count:Q", y=alt.Y("token:N", sort="-x", title=None)).properties(title="Most frequent tokens"), width="stretch")
            words = combined_text(df, prof["text_cols"]).str.split().str.len().rename("words").to_frame()
            st.altair_chart(alt.Chart(words).mark_bar(color="#3b6ea8").encode(
                x=alt.X("words:Q", bin=alt.Bin(maxbins=60), title="Words per row"), y=alt.Y("count()", title="Rows")).properties(
                title="Text length distribution"), width="stretch")
            art = pd.DataFrame([{"column": c, "artifact": k_, "rows": v} for c in prof["text_cols"]
                                for k_, v in prof["columns"][c].get("artifacts", {}).items()])
            if len(art) and art["rows"].sum():
                st.altair_chart(alt.Chart(art).mark_bar().encode(
                    x="rows:Q", y=alt.Y("artifact:N", title=None), color="column:N").properties(
                    title="Encoding and markup artifacts"), width="stretch")
    elif sub == "Correlation":
        corr = prof["correlation"]["matrix"]
        if corr is None:
            st.info("Not enough numeric or text features for a correlation matrix.")
        else:
            c = corr.reset_index().melt("index")
            st.altair_chart(alt.Chart(c).mark_rect().encode(
                x=alt.X("index:N", title=None), y=alt.Y("variable:N", title=None),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="r"),
                tooltip=["index", "variable", alt.Tooltip("value:Q", format=".2f")]).properties(
                height=460, title="Correlation of numeric and engineered text features"), width="stretch")

# ---------------------------------------------------------------- DL checks
if page == NAV[4]:
    texts = combined_text(df, prof["text_cols"]) if prof["text_cols"] else None
    st.markdown("#### Label noise (BiGRU classifier + confident learning)")
    if noise is None:
        st.info("Not run. Needs a label column with AG News style classes.")
    else:
        fl = noise[noise["flagged"]].sort_values("margin", ascending=False)
        k = st.columns(3)
        k[0].markdown(kpi("Rows judged", f"{len(noise):,}", f"{noise.attrs.get('skipped_short', 0):,} too short to judge"),
                      unsafe_allow_html=True)
        k[1].markdown(kpi("Suspected mislabels", f"{len(fl):,}", f"{100 * len(fl) / max(len(noise), 1):.2f}% of judged rows"),
                      unsafe_allow_html=True)
        k[2].markdown(kpi("Model agrees with labels", f"{100 * noise.attrs.get('accuracy_vs_given', 0):.1f}%", "on judged rows"),
                      unsafe_allow_html=True)
        st.caption(f"Per-class thresholds: {noise.attrs.get('thresholds')}")
        show = fl.head(200).copy()
        show.insert(1, "text", texts.loc[show["row"]].str.slice(0, 160).to_numpy())
        st.dataframe(show, width="stretch", hide_index=True,
                     column_config={"p_given": st.column_config.ProgressColumn("p(given)", min_value=0, max_value=1, format="%.3f"),
                                    "p_suggested": st.column_config.ProgressColumn("p(suggested)", min_value=0, max_value=1, format="%.3f")})
    st.markdown("#### Content anomalies (character autoencoder)")
    if anomalies is None:
        st.info("Not run.")
    else:
        thr = load_checker().ae_threshold
        k = st.columns(3)
        k[0].markdown(kpi("Rows checked", f"{len(anomalies):,}"), unsafe_allow_html=True)
        k[1].markdown(kpi("Anomalies", f"{int(anomalies['flagged'].sum()):,}"), unsafe_allow_html=True)
        k[2].markdown(kpi("z threshold", f"{thr:.2f}", "99th percentile of clean text"), unsafe_allow_html=True)
        hist = alt.Chart(anomalies.assign(z=anomalies["anomaly_z"].clip(-6, 30))).mark_bar(color="#3b6ea8").encode(
            x=alt.X("z:Q", bin=alt.Bin(maxbins=70), title="Length-adjusted anomaly z-score"), y=alt.Y("count()", title="Rows"))
        rule = alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(color="#b42318", strokeDash=[5, 3]).encode(x="t:Q")
        st.altair_chart((hist + rule).properties(title="Anomaly score (red line = threshold)"), width="stretch")
        top = anomalies.sort_values("anomaly_z", ascending=False).head(100).copy()
        top.insert(1, "text", texts.loc[top["row"]].str.slice(0, 160).to_numpy())
        st.dataframe(top, width="stretch", hide_index=True)

# ---------------------------------------------------------------- export
if page == NAV[5]:
    st.markdown("#### Audit report")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Report (HTML)", to_html(report), "data_quality_report.html", "text/html", icon=":material/web:", width="stretch")
    d2.download_button("Report (Markdown)", to_markdown(report), "data_quality_report.md", "text/markdown",
                       icon=":material/description:", width="stretch")
    d3.download_button("Issues (CSV)", issues_frame(issues).to_csv(index=False), "issues.csv", "text/csv",
                       icon=":material/list:", width="stretch")
    st.markdown("#### Recommended preprocessing plan")
    script = ["import numpy as np", "import pandas as pd", "", f"TEXT_COLS = {prof['text_cols']!r}",
              f"VALID_LABELS = {res['valid_labels']!r}", "df = pd.read_csv('your_data.csv')", ""]
    for n_, i in enumerate(report["pipeline"], 1):
        st.markdown(f"**{n_}. {i.title}**  \n{i.recommendation}")
        if i.code:
            st.code(i.code, language="python")
            script += [f"# {n_}. {i.title}", i.code, ""]
    st.download_button("Download plan as preprocess.py", "\n".join(script), "preprocess.py", "text/x-python",
                       icon=":material/code:")
