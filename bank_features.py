"""Feature engineering shared by the notebook and the Streamlit app.
(This file is WRITTEN by this notebook cell, so the notebook and the app always use exactly the same code.)"""
import numpy as np
import pandas as pd

# The 10 raw columns a client record must contain. (The target `deposit` is what we predict, never an input.)
RAW_FEATURES = ["age", "job", "balance", "housing", "loan", "contact", "month", "campaign", "pdays", "poutcome"]

# The allowed values of the text columns (they also fill the drop-down menus in the app)
CATEGORY_OPTIONS = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["unknown", "failure", "other", "success"],
}

# The columns the model sees AFTER feature engineering
NUMERIC_FEATURES = ["age", "pdays_clean", "balance_slog", "campaign_log", "was_contacted_before", "balance_negative", "has_both_loans"]
CATEGORICAL_FEATURES = ["job", "housing", "loan", "contact", "month", "poutcome", "age_group"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(frame):
    """Create model-ready features from the raw columns. Works row by row (stateless), so it is safe to run BEFORE the split."""
    out = pd.DataFrame(index=frame.index)                                    # start with an empty table, same rows

    out["age"] = frame["age"]                                                # keep as is
    out["job"] = frame["job"]
    out["housing"] = frame["housing"]
    out["loan"] = frame["loan"]
    out["contact"] = frame["contact"]
    out["month"] = frame["month"]
    out["poutcome"] = frame["poutcome"]

    out["was_contacted_before"] = (frame["pdays"] != -1).astype(int)         # True/False -> 1/0
    out["pdays_clean"] = frame["pdays"].where(frame["pdays"] != -1, np.nan)  # keep pdays only where it is a real value, else NaN
    out["balance_negative"] = (frame["balance"] < 0).astype(int)
    out["balance_slog"] = np.sign(frame["balance"]) * np.log1p(frame["balance"].abs())   # signed log
    out["campaign_log"] = np.log1p(frame["campaign"])                        # log(1 + x) is safe when x = 0
    out["age_group"] = pd.cut(frame["age"], bins=[0, 29, 39, 49, 59, 200],   # cut ages into bands
                              labels=["18-29", "30-39", "40-49", "50-59", "60+"]).astype(str)
    out["has_both_loans"] = ((frame["housing"] == "yes") & (frame["loan"] == "yes")).astype(int)
    return out
