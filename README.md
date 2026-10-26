# Bank Marketing Campaign — Finding the Most Prospective Clients

**The situation:** the bank cannot contact every client. Contact capacity, budget and customer goodwill are limited, and the campaign must become more cost-sustainable. So we build a model that identifies the clients most likely to subscribe to a term deposit, so that every contact counts. The cost that matters most is the **false positive** (contacting someone who says no), so the business KPI is **precision**.

**➡️ Start here: [`bank_marketing_portfolio.ipynb`](bank_marketing_portfolio.ipynb)** (already executed, all outputs included). A static copy is in [`reports/bank_marketing_portfolio.html`](reports/bank_marketing_portfolio.html).

## Results at a glance

| Model | CV ROC-AUC (train) | Test ROC-AUC |
|---|---|---|
| Baseline — majority class | 0.500 | 0.500 |
| Baseline — best single-feature lookup (`month`) | 0.642 | 0.681 |
| Decision tree | 0.742 | 0.752 |
| Logistic regression | 0.765 | 0.783 |
| Random forest | 0.779 | 0.786 |
| HistGradientBoosting | 0.779 | 0.787 |
| LightGBM | 0.778 | 0.785 |
| XGBoost | 0.781 | 0.789 |
| XGBoost (Optuna) | 0.780 | 0.790 |
| **LightGBM (Optuna) — selected** | 0.780 | **0.787** |
| YDF (Google) | 0.781 | 0.781 |

* The top models are **statistically tied** (differences ≪ the ±0.01 fold-to-fold spread). The best model is chosen with the **one-standard-error rule**: among the models tied with the top CV score, take the one that **overfits least** → **LightGBM (Optuna)**. Test scores are not used for the choice.
* At the default cut-off of 0.5: **precision 0.80, recall 0.60** → about **20 wasted contacts per 100** (52 when contacting at random). The top 30 % of clients ranked by the model have an **85 % hit rate** and cover 53 % of all prospects.
* **Optuna did not beat random search** (≤ 0.0015 AUC difference) — tuning is not the bottleneck, the data is.
* **XAI** (best model + YDF only): both agree (rank correlation 0.86) that `contact`, `month` and `poutcome` (previous success) drive predictions, followed by `balance`.
* **Checks:** no model is badly overfit; normality tests (no numeric feature is normal — irrelevant for our models); VIF (redundant columns are harmless for accuracy); missing data classified as MCAR / MAR / MNAR / structural.

## The Streamlit app

```bash
.venv\Scripts\streamlit run streamlit_app.py
```
(macOS/Linux: `.venv/bin/streamlit run streamlit_app.py`. The first load takes ~20 s while SHAP and LightGBM start up.)

* **👤 Single client** — fill in a form; get the probability of subscribing, a "contact / not prioritised" recommendation, and a **SHAP explanation** (grouped bar chart + waterfall).
* **📄 Bulk upload** — upload a CSV (template provided). Clients are validated, scored and ranked; flag prospects by **cut-off** and/or a **maximum number of clients** you can contact; download the results; see the **SHAP drivers** for the whole file and for any single client. If the file has a `deposit` column, the app also shows the actual precision / recall.
* The sidebar has the **cut-off slider** (default 0.5) and a model card.

The app loads **`models/best_model.parquet`**: a one-row Parquet file holding the full pipeline (raw client record → feature engineering → preprocessing → model) as a binary column, plus readable metadata (name, metrics, features, library versions). A `.joblib` copy is saved too.

> ⚠️ The model inside the Parquet file is a Python **pickle**. Only load model files you created yourself.

## What's in the notebook

1. Background (limited contact capacity) · 2. Aims & objectives · 3. Setup · 4. Load data
5. **First EDA — YData Profiling**
6. **Follow-up EDA** — missing data (MCAR / MAR / MNAR / structural, with an empirical check), duplicates, exploration, normality tests, statistical tests
7. **Feature engineering** (writes `bank_features.py`, shared with the app)
8. **Data-leakage checklist**, stratified split, **VIF check**
9. **Metrics** (ROC-AUC to select, precision as the business KPI) · 10. **Baselines**
11. **Models:** logistic regression → tree → random forest → HistGB / XGBoost / LightGBM → **YDF** → **Optuna**
12. Comparison, **best-model selection rule**, confusion matrices for every model, gains/lift, **which error matters most** (false positives) with cut-off trade-offs
13. **Over- and underfitting checks** (train vs CV vs test, learning curves, validation curves)
14. **XAI on the best model and YDF** — permutation importance, SHAP, partial dependence, YDF importance
15. **Saving the model (Parquet) and deploying in Streamlit**
16. Conclusions, limitations, next steps

Every code line is commented in plain English; every section opens with *what* and *why*.

## Repository contents

| File | Purpose |
|---|---|
| `bank_marketing_portfolio.ipynb` | The full analysis |
| `bank_features.py` | Feature engineering — **written by the notebook** (section 7), imported by the app |
| `bank_inference.py` | Helpers for the app: save/load Parquet model, validate uploads, predict, SHAP |
| `streamlit_app.py` | The Streamlit app |
| `models/best_model.parquet`, `models/best_model.joblib` | The deployable model |
| `reports/` | YData Profiling report, HTML export of the notebook, model comparison CSV |
| `requirements.txt` | Pinned library versions |

## Reproduce it

`ydata-profiling` does not yet support Python 3.14, so the project uses a **Python 3.12** virtual environment (`.venv`, created with [uv](https://docs.astral.sh/uv/)):

```bash
python -m pip install uv
python -m uv venv .venv --python 3.12
python -m uv pip install --python .venv/Scripts/python.exe -r requirements.txt
python -m uv pip install --python .venv/Scripts/python.exe "setuptools<81"
.venv/Scripts/python.exe -m ipykernel install --user --name bank-marketing --display-name "Python 3.12 (bank-marketing)"
```

### Data

The raw dataset (`data_bank_marketing_campaign.csv`, 7,813 rows × 11 columns) and the original task brief are **not included** in this repository. To re-run the notebook, place the CSV in the project folder next to the notebook. The **Streamlit app does not need it**: the trained model (`models/`) and `bank_features.py` are included, so you can run the app straight after installing the requirements.

### Re-running the notebook

Open the notebook, select the **Python 3.12 (bank-marketing)** kernel and *Run All* (~6 minutes). Results are seeded (`RANDOM_STATE = 42`). Running it regenerates `bank_features.py` and the files in `models/`.

> `setuptools<81` is needed because `ydata-profiling` still imports `pkg_resources`.

## Honest limitations

* **`campaign` and `month` are recorded after the fact** (`campaign` = contacts made so far in this campaign; `month` = the *last* contact month). With no date column there is no time-based validation, so the model has never been tested on a *future* campaign.
* **`contact = unknown` is tied to how contacts were logged** (mostly May–June); the model's use of it only holds if logging stays consistent.
* A single test split (≈ ±0.015 AUC margin), no fairness review (the model uses `age` and `job`), and probabilities that are only approximately calibrated. Details and next steps are in section 16 of the notebook.
