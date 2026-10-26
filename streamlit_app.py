"""Prospective-client finder — Streamlit app.

Run with:   streamlit run streamlit_app.py

Loads the best model from models/best_model.parquet (created by the notebook, section 15) and offers
  1. a single-client prediction from a form, and
  2. a bulk upload of many clients,
both with a SHAP explanation of why each client got its score.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                   # draw charts off-screen; Streamlit shows the image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score

from bank_features import CATEGORY_OPTIONS, RAW_FEATURES
from bank_inference import (explain, group_shap, load_model_parquet, make_explainer, predict_proba,
                            probability_from_logodds, validate_records)

MODEL_PATH = Path(__file__).parent / "models" / "best_model.parquet"
ORANGE, GREY = "#e0761a", "#8c9aa8"
MAX_ROWS_FOR_SHAP = 5000                                 # keep the bulk SHAP summary fast

st.set_page_config(page_title="Prospective client finder", page_icon="🏦", layout="wide")


# ------------------------------------------------------------------------------------------------
# Loading (cached: done once, not on every click)
# ------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model...")
def load_artifacts():
    pipeline, meta = load_model_parquet(MODEL_PATH)
    return pipeline, meta, make_explainer(pipeline)


@st.cache_data(show_spinner="Computing SHAP values for the whole file...")
def global_shap(_pipeline, _explainer, records: pd.DataFrame) -> pd.DataFrame:
    """SHAP values grouped by original column for (up to) MAX_ROWS_FOR_SHAP clients."""
    sample = records.head(MAX_ROWS_FOR_SHAP)
    explanation, encoded = explain(_pipeline, sample, _explainer)
    grouped = group_shap(explanation, encoded.columns)
    grouped.index = sample.index
    return grouped


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}. Run the notebook (section 15) to create it.")
    st.stop()
pipeline, meta, explainer = load_artifacts()


# ------------------------------------------------------------------------------------------------
# Small drawing helpers
# ------------------------------------------------------------------------------------------------
def draw_grouped_bars(contributions: pd.Series, title: str):
    """Horizontal bars: each original column's push on one client's score (right = towards 'subscribe')."""
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top.index, top.values, color=[ORANGE if v > 0 else GREY for v in top.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Push on the score (log-odds)\n← towards 'no'   |   towards 'yes' →")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def explain_one(records: pd.DataFrame):
    """Show the SHAP explanation of the FIRST row of `records`."""
    explanation, encoded = explain(pipeline, records.head(1), explainer)
    grouped = group_shap(explanation, encoded.columns).iloc[0]
    base = float(np.ravel(explanation.base_values)[0])
    final = base + float(explanation.values[0].sum())

    st.caption(
        f"SHAP starts from the **average client** (score ≈ {probability_from_logodds(base):.0%}) and adds up how much each "
        f"column pushes this client up or down, ending at **{probability_from_logodds(final):.0%}**.")
    left, right = st.columns(2)
    plt.close("all")                                                     # start from a clean slate (Streamlit reruns this script often)
    with left:
        st.pyplot(draw_grouped_bars(grouped, "Which columns drove this score?"), clear_figure=True)
    plt.close("all")                                                     # so the SHAP waterfall gets its own fresh figure
    with right:
        shap.plots.waterfall(explanation[0], max_display=12, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    plt.close("all")
    with st.expander("How to read these charts"):
        st.markdown(
            "* **Orange / red bars** push the client towards *subscribing*; **grey / blue bars** push towards *not subscribing*.\n"
            "* The left chart adds up the one-hot columns (e.g. every `month_*` column becomes just `month`) so it is easier to read.\n"
            "* The right chart is the detailed SHAP waterfall, one bar per encoded column (e.g. `month_may`).\n"
            "* Units are *log-odds*, the model's internal scale; larger bars = stronger influence. SHAP explains what the "
            "**model** uses — it does not prove that changing a column would change the client's decision.")


def decision_text(probability: float, threshold: float) -> str:
    return "✅ Prospective client — worth contacting" if probability >= threshold else "⏸️ Not prioritised at this cut-off"


# ------------------------------------------------------------------------------------------------
# Sidebar: model card + the operating point
# ------------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    st.markdown(f"**{meta['model_name']}**")
    c1, c2 = st.columns(2)
    c1.metric("Test ROC-AUC", f"{meta['test_roc_auc']:.3f}")
    c2.metric("Precision @0.5", f"{meta['test_precision_at_0.5']:.2f}")
    st.caption(f"Recall @0.5: {meta['test_recall_at_0.5']:.2f} · trained on {meta['trained_on_rows']:,} clients")

    st.header("Operating point")
    threshold = st.slider(
        "Cut-off: contact clients with P(subscribe) ≥", 0.10, 0.95, float(meta["default_threshold"]), 0.05,
        help="Higher = fewer, surer contacts (fewer wasted contacts, but you reach fewer prospects). "
             "In our tests 0.5 gave 80 % precision and 60 % recall.")
    st.caption("Higher cut-off → fewer wasted contacts, but fewer prospects reached.")

    with st.expander("⚠️ Please read: limits of this model"):
        st.markdown(
            "* Trained on one historical campaign; it has **no date information**, so it was never tested on a *future* campaign.\n"
            "* `campaign` means *contacts made to this client so far in the campaign*; `month` is the month of the "
            "**last** contact. Both were recorded after the fact in the training data.\n"
            "* `contact = unknown` is linked to how contacts were logged in May–June. Keep recording practice consistent.\n"
            "* The score is a *ranking aid*, not a guarantee. Check fairness and regulations before using it to decide "
            "whom to contact.")

st.title("🏦 Prospective client finder")
st.markdown(
    "Our contact capacity is limited, so we want to spend it on the clients **most likely to subscribe to a term deposit**. "
    "This app scores clients with the model built in the notebook and shows **why** each client got its score.")

tab_single, tab_bulk = st.tabs(["👤 Single client", "📄 Bulk upload"])


# ------------------------------------------------------------------------------------------------
# Tab 1: single client
# ------------------------------------------------------------------------------------------------
with tab_single:
    st.subheader("Describe one client")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 100, 40)
        job = st.selectbox("Job", CATEGORY_OPTIONS["job"], index=CATEGORY_OPTIONS["job"].index("management"))
        balance = st.number_input("Average yearly balance", -10000, 150000, 500, step=100)
    with col2:
        housing = st.radio("Housing loan?", CATEGORY_OPTIONS["housing"], horizontal=True)
        loan = st.radio("Personal loan?", CATEGORY_OPTIONS["loan"], horizontal=True)
        contact = st.selectbox("How is the client contacted?", CATEGORY_OPTIONS["contact"])
        month = st.selectbox("Month of (last) contact", CATEGORY_OPTIONS["month"], index=CATEGORY_OPTIONS["month"].index("may"))
    with col3:
        campaign = st.number_input("Contacts made to this client in this campaign (so far)", 1, 60, 2)
        contacted_before = st.checkbox("Contacted in a previous campaign?", value=False)
        pdays = st.number_input("Days since the previous contact", 1, 900, 90, disabled=not contacted_before)
        poutcome = st.selectbox("Outcome of the previous campaign", CATEGORY_OPTIONS["poutcome"],
                                index=CATEGORY_OPTIONS["poutcome"].index("failure"), disabled=not contacted_before)

    record = pd.DataFrame([{
        "age": age, "job": job, "balance": balance, "housing": housing, "loan": loan, "contact": contact, "month": month,
        "campaign": campaign,
        "pdays": int(pdays) if contacted_before else -1,                      # -1 = never contacted before
        "poutcome": poutcome if contacted_before else "unknown",
    }])
    clean_one, report_one = validate_records(record)
    for message in report_one["errors"] + report_one["warnings"]:
        st.warning(message)

    if len(clean_one):
        probability = float(predict_proba(pipeline, clean_one)[0])
        st.divider()
        m1, m2 = st.columns([1, 2])
        m1.metric("Probability of subscribing", f"{probability:.0%}")
        with m2:
            (st.success if probability >= threshold else st.info)(decision_text(probability, threshold))
            st.progress(min(max(probability, 0.0), 1.0))
            st.caption(f"Cut-off in use: {threshold:.2f} (change it in the sidebar). "
                       f"Of clients scoring around {probability:.0%}, about that share subscribed in our tests.")
        st.subheader("Why this score?")
        explain_one(clean_one)


# ------------------------------------------------------------------------------------------------
# Tab 2: bulk upload
# ------------------------------------------------------------------------------------------------
with tab_bulk:
    st.subheader("Score a whole list of clients")
    st.markdown(
        "Upload a **CSV** with one row per client and these columns: "
        + ", ".join(f"`{c}`" for c in RAW_FEATURES)
        + ". A `deposit` column (yes/no) is optional — if present, the app also checks how well the model did.")

    example = pd.DataFrame([
        {"age": 34, "job": "student", "balance": 2500, "housing": "no", "loan": "no", "contact": "cellular", "month": "mar",
         "campaign": 1, "pdays": 120, "poutcome": "success"},
        {"age": 45, "job": "blue-collar", "balance": 120, "housing": "yes", "loan": "no", "contact": "unknown", "month": "may",
         "campaign": 4, "pdays": -1, "poutcome": "unknown"},
        {"age": 67, "job": "retired", "balance": 8000, "housing": "no", "loan": "no", "contact": "cellular", "month": "oct",
         "campaign": 2, "pdays": -1, "poutcome": "unknown"},
    ])
    st.download_button("⬇️ Download a CSV template", example.to_csv(index=False), "client_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    max_clients = st.number_input("Maximum number of clients we can contact (0 = no limit)", 0, 1_000_000, 0, step=10,
                                  help="Capacity rule: even if more clients pass the cut-off, only the best-ranked ones are selected.")

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, sep=None, engine="python")           # sep=None: detects comma / semicolon
        except Exception as error:                                           # unreadable file
            st.error(f"Could not read the file as a CSV: {error}")
            st.stop()

        records, report = validate_records(raw)
        for message in report["errors"]:
            st.error(message)
        for message in report["warnings"]:
            st.warning(message)
        if len(report["dropped"]):
            with st.expander(f"Rows that were skipped ({len(report['dropped'])})"):
                st.dataframe(report["dropped"], hide_index=True)

        if len(records) == 0:
            st.stop()

        # ---- score and rank ----
        records = records.copy()
        records["P(subscribe)"] = predict_proba(pipeline, records)
        records["rank"] = records["P(subscribe)"].rank(ascending=False, method="first").astype(int)
        capacity_ok = (records["rank"] <= max_clients) if max_clients > 0 else True
        records["contact?"] = np.where((records["P(subscribe)"] >= threshold) & capacity_ok, "yes", "no")
        records.insert(0, "row in file", records.index + 1)
        ranked = records.sort_values("rank")
        chosen = ranked[ranked["contact?"] == "yes"]

        # ---- headline numbers ----
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clients scored", f"{len(records):,}")
        k2.metric("Selected for contact", f"{len(chosen):,}", f"{len(chosen) / len(records):.0%} of the list")
        expected_hit = chosen["P(subscribe)"].mean() if len(chosen) else float("nan")
        k3.metric("Expected hit rate", "—" if not len(chosen) else f"{expected_hit:.0%}",
                  help="Average predicted probability among the selected clients.")
        k4.metric("Expected wasted contacts", "—" if not len(chosen) else f"{len(chosen) * (1 - expected_hit):.0f}",
                  help="Selected clients × (1 − expected hit rate).")

        # ---- results table + download ----
        st.markdown("#### Ranked clients")
        st.dataframe(
            ranked.drop(columns=["deposit"], errors="ignore"), hide_index=True, height=380,
            column_config={"P(subscribe)": st.column_config.ProgressColumn("P(subscribe)", min_value=0, max_value=1, format="%.2f")})
        st.download_button("⬇️ Download all results (CSV)", ranked.to_csv(index=False), "scored_clients.csv", "text/csv")

        # ---- optional: if the file has the true answers, check the model ----
        if "deposit" in records.columns:
            truth = records["deposit"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
            if truth.notna().all() and truth.nunique() == 2:
                with st.expander("📊 The file contains the true outcomes — how did the model do?", expanded=True):
                    picked = (records["contact?"] == "yes").to_numpy()
                    tp = int((picked & (truth == 1)).sum()); fp = int((picked & (truth == 0)).sum())
                    fn = int((~picked & (truth == 1)).sum()); tn = int((~picked & (truth == 0)).sum())
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Actual hit rate (precision)", f"{tp / max(tp + fp, 1):.0%}")
                    e2.metric("Prospects reached (recall)", f"{tp / max(tp + fn, 1):.0%}")
                    e3.metric("Wasted contacts (FP)", f"{fp:,}")
                    e4.metric("ROC-AUC", f"{roc_auc_score(truth, records['P(subscribe)']):.3f}")
                    st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["actually no", "actually yes"],
                                              columns=["not contacted", "contacted"]))
                    st.caption("⚠️ If this file contains clients the model was trained on, these numbers are optimistic. "
                               "Only clients the model has never seen give an honest estimate.")

        # ---- SHAP: global drivers + one client ----
        st.markdown("#### Why? SHAP explanations")
        grouped = global_shap(pipeline, explainer, records[RAW_FEATURES])
        if len(records) > MAX_ROWS_FOR_SHAP:
            st.caption(f"The overall summary uses the first {MAX_ROWS_FOR_SHAP:,} clients.")
        g_left, g_right = st.columns(2)
        with g_left:
            mean_abs = grouped.abs().mean().sort_values().tail(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(mean_abs.index, mean_abs.values, color=ORANGE)
            ax.set_xlabel("Average |push| on the score (log-odds)")
            ax.set_title("What drives the scores in this file?", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        with g_right:
            st.markdown("**Explain one client**")
            default_row = int(ranked.iloc[0]["row in file"])
            row_number = st.number_input("Row number in your file", 1, int(records["row in file"].max()), default_row,
                                         help="Defaults to the best-ranked client.")
            match = records[records["row in file"] == row_number]
            if len(match):
                client = match.iloc[0]
                st.write(f"Row **{row_number}** — rank **{int(client['rank'])}** of {len(records):,}, "
                         f"P(subscribe) = **{client['P(subscribe)']:.0%}** → "
                         f"{decision_text(client['P(subscribe)'], threshold)}")
                st.dataframe(match[RAW_FEATURES], hide_index=True)
            else:
                st.info("That row was skipped during validation (see the list above).")
        if len(match):
            explain_one(match[RAW_FEATURES])
    else:
        st.info("Upload a file to get started — or download the template above to see the expected format.")
