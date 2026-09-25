"""Streamlit demo: upload a dirty CSV, watch it get cleaned, see the honest model audit
and a transparent Readiness Index. Run: streamlit run app/app.py"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import CATEGORICAL_VOCAB, NUMERIC_COLS, TARGET, clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Career Readiness: Pandas & Co", layout="wide")
st.title("Career Readiness Datathon: Pandas & Co")
st.caption("Analytrix TCET, Problem Statement 3. 50,000 rows, 10 corruption types, one honest answer.")


@st.cache_data
def load_sample():
    return pd.read_csv(ROOT / "data/raw/ps3.csv", dtype=str, keep_default_na=False)


@st.cache_resource
def load_model():
    p = ROOT / "models/model.pkl"
    return joblib.load(p) if p.exists() else None


def readiness_index(df: pd.DataFrame) -> pd.Series:
    """Transparent 0-100 composite. Missing inputs take the column median (or a neutral default
    for a single profile). ponytail: fixed weights, tune with domain experts if used for real."""
    defaults = {"Technical_Skill_Score": 70, "Soft_Skill_Score": 70, "Domain_Knowledge_Score": 70,
                "Years_of_Experience": 5, "Certification_Count": 4, "Training_Hours_Last_Year": 250,
                "Skill_Gap_Score": 50}
    df = df[list(defaults)].apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median()).fillna(pd.Series(defaults))
    skills = df[["Technical_Skill_Score", "Soft_Skill_Score", "Domain_Knowledge_Score"]].mean(axis=1)
    exp = (df["Years_of_Experience"].clip(0, 20) / 20 * 100)
    certs = (df["Certification_Count"].clip(0, 8) / 8 * 100)
    train = (df["Training_Hours_Last_Year"].clip(0, 500) / 500 * 100)
    gap = 100 - df["Skill_Gap_Score"].clip(0, 100)
    return (0.4 * skills + 0.2 * exp + 0.1 * certs + 0.1 * train + 0.2 * gap).round(1)


up = st.sidebar.file_uploader("Upload a dirty CSV (same columns as PS3)", type="csv")
raw = pd.read_csv(up, dtype=str, keep_default_na=False) if up else load_sample()
required = set(NUMERIC_COLS) | set(CATEGORICAL_VOCAB) - {TARGET}
missing_cols = required - set(raw.columns)
if missing_cols:
    st.error(f"Missing columns: {sorted(missing_cols)}")
    st.stop()

df, report = clean(raw, drop_bad_target=False)
tab1, tab2, tab3, tab4 = st.tabs(["1. Cleaning", "2. Insights", "3. Model audit", "4. Readiness Index"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(raw):,}")
    c2.metric("Values repaired", f"{int(report.drop(columns=['missing', 'unknown']).to_numpy().sum()):,}")
    c3.metric("Unparseable left", int(report["unknown"].sum()))
    st.subheader("Corruption report (count of fixes per column)")
    st.dataframe(report, use_container_width=True)
    st.subheader("Before and after")
    n = st.slider("rows to preview", 5, 50, 10)
    st.write("Raw"); st.dataframe(raw.head(n), use_container_width=True)
    st.write("Cleaned"); st.dataframe(df.head(n), use_container_width=True)
    st.download_button("Download cleaned CSV", df.to_csv(index=False).encode(), "cleaned.csv", "text/csv")

with tab2:
    for f, cap in [("01_corruption_heatmap.png", "Every column is hit by 8 to 30 percent per corruption type."),
                   ("02_recovery.png", "Cleaning lifts usable values from roughly 45 percent to 80 percent per column."),
                   ("04_distributions.png", "All numeric features are uniform: this is synthetic data."),
                   ("05_correlation.png", "No two features are correlated above 0.02.")]:
        p = ROOT / "reports" / f
        if p.exists():
            st.image(str(p), caption=cap, use_column_width=True)

with tab3:
    mp = ROOT / "reports/metrics.json"
    if mp.exists():
        m = json.load(open(mp))
        st.markdown(f"**Chance accuracy:** {m['chance_accuracy']:.3f}  ·  "
                    f"**Shuffled-label accuracy:** {m['permutation_test']['mean']:.3f}")
        st.dataframe(pd.DataFrame(m["models"]).T.round(4), use_container_width=True)
        st.dataframe(pd.DataFrame(m["chi_square"]).T.round(3), use_container_width=True)
        st.warning("Every model equals chance and every chi-square p-value exceeds 0.36. "
                   "The label in this dataset is statistically independent of all features. "
                   "We report that rather than a leaked or overfit score.")
        p = ROOT / "reports/07_model_vs_chance.png"
        if p.exists():
            st.image(str(p), use_column_width=True)
    model = load_model()
    if model is not None:
        st.subheader("Model predictions on uploaded data (for completeness)")
        X = df.drop(columns=[c for c in ["ID", TARGET] if c in df])
        pred = pd.DataFrame(model.predict_proba(X), columns=model.classes_).round(3)
        st.dataframe(pd.concat([df[["ID"]], pred], axis=1).head(20), use_container_width=True)

with tab4:
    st.markdown("Because the label is noise, we offer a **transparent Readiness Index** built only from "
                "the cleaned inputs: 40% skills, 20% experience, 20% closed skill gap, 10% certifications, "
                "10% training hours.")
    df["Readiness_Index"] = readiness_index(df)
    st.dataframe(df[["ID", "Current_or_Target_Role", "Industry", "Readiness_Index"]].head(20),
                 use_container_width=True)
    st.bar_chart(df.groupby("Current_or_Target_Role")["Readiness_Index"].mean().sort_values())
    st.subheader("Try your own profile")
    cols = st.columns(4)
    me = {
        "Technical_Skill_Score": cols[0].slider("Technical", 40, 100, 70),
        "Soft_Skill_Score": cols[1].slider("Soft skills", 40, 100, 70),
        "Domain_Knowledge_Score": cols[2].slider("Domain", 40, 100, 70),
        "Skill_Gap_Score": cols[3].slider("Skill gap", 0, 100, 40),
        "Years_of_Experience": cols[0].slider("Years experience", 0, 20, 3),
        "Certification_Count": cols[1].slider("Certifications", 0, 8, 2),
        "Training_Hours_Last_Year": cols[2].slider("Training hours", 0, 500, 100),
    }
    st.metric("Your Readiness Index", readiness_index(pd.DataFrame([me])).iloc[0])
