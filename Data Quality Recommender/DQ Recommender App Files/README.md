# DQ-Recommender

Deep learning assisted data quality auditor for text datasets. It profiles a CSV, runs two
PyTorch models trained from scratch on AG News, and writes a data quality report with
severity-ranked issues, a 0 to 100 quality score and an ordered preprocessing plan. A
**preprocessing agent** then applies the fixes step by step (observe, plan, act, verify),
explains why each step helps, and measures the effect on a downstream model.

Dataset: [AG News Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

## What is in this folder

| Path | What it is |
|---|---|
| `DQ_Recommender_AG_News.ipynb` | The project notebook (sections 1 to 13 of the project instructions), executed with outputs |
| `DQ_Recommender_AG_News.pdf` | PDF export of the notebook (for D2L) |
| `DQ_Recommender_AG_News.html` | HTML export of the notebook |
| `app.py` | Streamlit demo application |
| `run_app.bat` | Double-click to start the app on Windows |
| `dq/` | The package shared by notebook and app: cleaning, models, profiler, DL checks, recommender, cleaner, agent (`dq/agent.py`) |
| `artifacts/` | Trained weights (`classifier.pt`, `autoencoder.pt`), vocabulary and calibration (`config.json`) |
| `sample_data/` | Demo datasets: AG News test set with injected issues (+ ground truth), the original test set, a 5,000-row clean reference set used to measure downstream impact; example reports |
| `runs/` | TensorBoard logs of the classifier training (`tensorboard --logdir runs`) |

## Run the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501 (on Windows you can also double-click `run_app.bat`). No
GPU is needed. In the sidebar, pick one of the samples or upload your own CSV, check the
text and label columns, and press **Run data quality audit**. Then open **Preprocessing
agent** and press **Run preprocessing agent** to clean the data automatically (about 30 s on
the demo; about 70 s with the per-step contribution measurement switched on).

## Re-run the notebook

Open `DQ_Recommender_AG_News.ipynb` in Jupyter and run all cells. The data is downloaded
with `kagglehub`. If that fails, put `train.csv` and `test.csv` from Kaggle into `data/`.
A full run takes about 15 to 20 minutes on a laptop RTX 3050. It rewrites `dq/`, `app.py`,
`artifacts/` and `sample_data/`.

## Results (AG News test set)

| | Value |
|---|---|
| BiGRU classifier test accuracy | 91.97% (TF-IDF + logistic regression baseline: 92.32%) |
| Label-flip detection on the benchmark | recall 91.3%, ROC-AUC 0.975 |
| Junk rows (gibberish, typos, off-domain) detected | 100% (autoencoder + OOV rule), 1.1% autoencoder false alarms |
| Rule-based issues detected (missing, blanks, duplicates, HTML, truncation, invalid labels) | 100% |
| Benchmark quality score before / after applying the recommendations | 86.1 / 98.8 |
| Preprocessing agent on the benchmark: score / rows | 86.2 -> 99.4 (after a fresh re-audit) / 7,866 -> 6,786 |
| Quick TF-IDF model trained on the benchmark before / after the agent | 87.60% / 88.52% accuracy on 5,000 clean reference rows |
