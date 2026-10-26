"""Helpers shared by the notebook and the Streamlit app.

* save / load the trained model as a Parquet file
* validate customer records typed in or uploaded by a user
* predict the probability of subscribing
* explain a prediction with SHAP (grouped back to the original columns)

This file is hand-written (the notebook only imports it). It relies on ``bank_features.py``,
which the notebook writes (section 7).
"""
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bank_features import CATEGORICAL_FEATURES, CATEGORY_OPTIONS, NUMERIC_FEATURES, RAW_FEATURES

NUMERIC_RAW = ["age", "balance", "campaign", "pdays"]            # raw columns that must be numbers
TEXT_RAW = [c for c in RAW_FEATURES if c not in NUMERIC_RAW]       # raw columns that must be text categories


# ---------------------------------------------------------------------------------------------
# 1. Saving and loading the model as a Parquet file
# ---------------------------------------------------------------------------------------------
def save_model_parquet(pipeline, path, metadata):
    """Store the fitted pipeline in ONE row of a Parquet file.

    Parquet is a table format, so the model itself is stored as a binary column (``model_bytes``).
    The other columns hold human-readable information about the model (name, metrics, features...).
    """
    row = {
        "model_name": metadata["model_name"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata_json": json.dumps(metadata, default=str),
        "model_bytes": pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL),
    }
    pd.DataFrame([row]).to_parquet(path, index=False)


def load_model_parquet(path):
    """Return (pipeline, metadata). SECURITY: the model is a pickle - only load files you created yourself."""
    row = pd.read_parquet(path).iloc[0]
    return pickle.loads(row["model_bytes"]), json.loads(row["metadata_json"])


# ---------------------------------------------------------------------------------------------
# 2. Checking customer records before they reach the model
# ---------------------------------------------------------------------------------------------
def validate_records(frame):
    """Clean an uploaded table and report problems.

    Returns ``(clean, report)`` where ``clean`` holds only the rows that can be scored (columns in the
    right order, correct types) and ``report`` is a dict with ``errors`` (fatal), ``warnings`` and
    ``dropped`` (a table of rows that were skipped and why).
    """
    report = {"errors": [], "warnings": [], "dropped": pd.DataFrame(columns=["row", "reason"])}
    frame = frame.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]          # tolerate "Age", " age " ...

    missing_cols = [c for c in RAW_FEATURES if c not in frame.columns]
    if missing_cols:
        report["errors"].append(f"Missing required column(s): {', '.join(missing_cols)}")
        return pd.DataFrame(columns=RAW_FEATURES), report

    extra = [c for c in frame.columns if c not in RAW_FEATURES and c != "deposit"]
    if extra:
        report["warnings"].append(f"Ignoring unused column(s): {', '.join(extra)}")

    keep = frame[RAW_FEATURES + (["deposit"] if "deposit" in frame.columns else [])].copy()
    for col in TEXT_RAW:                                                       # "Cellular " -> "cellular"
        keep[col] = keep[col].astype("string").str.strip().str.lower()
    for col in NUMERIC_RAW:                                                    # "12abc" -> NaN
        keep[col] = pd.to_numeric(keep[col], errors="coerce")

    bad = keep[RAW_FEATURES].isna().any(axis=1)                                # any missing/unparseable value
    problems = []
    for idx in keep.index[bad]:
        empty = [c for c in RAW_FEATURES if pd.isna(keep.at[idx, c])]
        problems.append({"row": int(idx) + 1, "reason": "missing or invalid value in: " + ", ".join(empty)})
    if problems:
        report["dropped"] = pd.DataFrame(problems)
        report["warnings"].append(f"{len(problems)} row(s) skipped because of missing/invalid values.")
    clean = keep.loc[~bad].copy()

    # Soft checks: the row is still scored, but the user is told
    for col, options in CATEGORY_OPTIONS.items():
        unseen = sorted(set(clean[col].dropna().unique()) - set(options))
        if unseen:
            report["warnings"].append(
                f"`{col}` contains value(s) the model has never seen: {', '.join(map(str, unseen))}. "
                "They are scored as 'none of the known categories', so treat those predictions with care.")
    if ((clean["age"] < 18) | (clean["age"] > 100)).any():
        report["warnings"].append("Some ages are outside 18-100 - the model was trained on adults only.")
    if (clean["campaign"] < 1).any():
        report["warnings"].append("`campaign` should be at least 1 (number of contacts in this campaign).")
    if (clean["pdays"] < -1).any():
        report["warnings"].append("`pdays` should be -1 (never contacted before) or a positive number of days.")
    inconsistent = ((clean["pdays"] == -1) & (clean["poutcome"] != "unknown")).sum()
    if inconsistent:
        report["warnings"].append(f"{inconsistent} row(s) have pdays = -1 (never contacted) but a known previous outcome.")

    for col in NUMERIC_RAW:
        clean[col] = clean[col].astype(float)
    for col in TEXT_RAW:
        clean[col] = clean[col].astype(str)
    return clean, report


# ---------------------------------------------------------------------------------------------
# 3. Predicting
# ---------------------------------------------------------------------------------------------
def predict_proba(pipeline, raw_records):
    """Probability that each customer subscribes (the 'yes' class)."""
    return pipeline.predict_proba(raw_records[RAW_FEATURES])[:, 1]


# ---------------------------------------------------------------------------------------------
# 4. Explaining with SHAP
# ---------------------------------------------------------------------------------------------
def make_explainer(pipeline):
    import shap                                                              # imported here: heavy
    return shap.TreeExplainer(pipeline.named_steps["model"])


def explain(pipeline, raw_records, explainer=None):
    """SHAP values for each customer. Returns (shap.Explanation, encoded_features_table).

    The values are in log-odds: positive pushes towards 'yes', negative towards 'no'.
    """
    explainer = explainer or make_explainer(pipeline)
    encoded = pipeline[:-1].transform(raw_records[RAW_FEATURES])             # everything except the final model
    explanation = explainer(encoded)
    if explanation.values.ndim == 3:                                         # some versions return one slice per class
        explanation = explanation[..., 1]
    return explanation, encoded


def original_feature(encoded_name):
    """'month_may' -> 'month', 'age_group_30-39' -> 'age_group', 'balance_slog' -> 'balance_slog'."""
    for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):          # longest prefix first (age_group before age)
        if encoded_name.startswith(col + "_"):
            return col
    return encoded_name


def group_shap(explanation, feature_names):
    """Sum the SHAP values of one-hot columns back into their original column. Returns a DataFrame (rows x features)."""
    groups = [original_feature(n) for n in feature_names]
    values = pd.DataFrame(explanation.values, columns=groups)
    return values.T.groupby(level=0).sum().T


def probability_from_logodds(x):
    return 1.0 / (1.0 + np.exp(-x))
