"""Train + audit. Answers one question honestly: can Career_Readiness_Level be predicted?

Runs cross-validated LogReg and HistGradientBoosting, a permutation test on the
label, and chi-square tests of each categorical against the label. Saves the
best pipeline to models/model.pkl so the app can use it, and metrics to
reports/metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocess import CATEGORICAL_VOCAB, NUMERIC_COLS, TARGET, clean

CAT_COLS = [c for c in CATEGORICAL_VOCAB if c != TARGET]
SEED = 42


def build_pipeline(kind: str, numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    clf = (LogisticRegression(max_iter=1000) if kind == "logreg"
           else HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED))
    return Pipeline([("pre", pre), ("clf", clf)])


def cv_acc(pipe, X, y) -> tuple[float, float]:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    s = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    return float(s.mean()), float(s.std())


def main(src="data/raw/ps3.csv"):
    df, _ = clean(pd.read_csv(src, dtype=str, keep_default_na=False))
    X, y = df.drop(columns=["ID", TARGET]), df[TARGET]
    chance = float(y.value_counts(normalize=True).max())
    out = {"n_rows": int(len(df)), "chance_accuracy": chance, "models": {}, "chi_square": {}}

    for kind in ("logreg", "hgb"):
        for label, numeric in (("all_features", NUMERIC_COLS),
                               ("without_readiness_score", [c for c in NUMERIC_COLS if c != "Readiness_Score"])):
            m, s = cv_acc(build_pipeline(kind, numeric), X[numeric + CAT_COLS], y)
            out["models"][f"{kind}/{label}"] = {"cv_accuracy": m, "cv_std": s}
            print(f"{kind:7s} {label:25s} acc={m:.4f} +/- {s:.4f}")

    # permutation test: does shuffling the label change accuracy? if not, there is no signal.
    rng = np.random.default_rng(SEED)
    perm = [cv_acc(build_pipeline("hgb", NUMERIC_COLS), X, pd.Series(rng.permutation(y), index=y.index))[0]
            for _ in range(20)]
    out["permutation_test"] = {"shuffled_label_accuracies": perm, "mean": float(np.mean(perm))}
    print(f"shuffled-label acc (20 runs): mean={np.mean(perm):.4f}")

    for c in CAT_COLS:
        chi2, p, _, _ = chi2_contingency(pd.crosstab(df[c], y))
        out["chi_square"][c] = {"chi2": float(chi2), "p_value": float(p)}
        print(f"chi2 {c:24s} p={p:.3f}")

    final = build_pipeline("hgb", NUMERIC_COLS).fit(X, y)
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(final, "models/model.pkl")
    json.dump(out, open("reports/metrics.json", "w"), indent=2)
    print("saved models/model.pkl and reports/metrics.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
