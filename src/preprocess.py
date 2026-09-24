"""Cleaning pipeline for the PS3 career-readiness dataset.

The raw file is corrupted in eight ways; each has one small fixer below and
``clean()`` composes them and counts every fix it applied.
"""
from __future__ import annotations

import base64
import binascii
import json
import re

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "nan", "null", "missing", "?", "none", "na", "n/a"}

CATEGORICAL_VOCAB = {
    "Education_Level": ["High School", "Diploma", "Bachelor", "Master", "PhD"],
    "Current_or_Target_Role": [
        "Software Engineer", "Cybersecurity Analyst", "System Admin", "ML Engineer",
        "Data Analyst", "Cloud Engineer", "Business Analyst",
    ],
    "Employment_Status": ["Employed", "Student", "Unemployed"],
    "Industry": ["Healthcare", "Education", "IT", "Manufacturing", "Finance", "Retail"],
    "Career_Readiness_Level": ["Low", "Medium", "High"],
}
NUMERIC_COLS = [
    "Age", "Years_of_Experience", "Technical_Skill_Score", "Soft_Skill_Score",
    "Domain_Knowledge_Score", "Certification_Count", "Training_Hours_Last_Year",
    "Skill_Gap_Score", "Readiness_Score", "Annual_Salary_USD",
]
TARGET = "Career_Readiness_Level"
REPORT_COLS = ["missing", "json", "range", "prefix", "currency", "base64", "reversed", "hex", "unknown"]

_RANGE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)$")
_B64 = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _is_missing(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)


def _unwrap_json(v: str) -> str:
    if v.startswith("{"):
        try:
            return str(json.loads(v)["v"])
        except (ValueError, KeyError, TypeError):
            pass
    return v


def _try_b64(v: str) -> str | None:
    if len(v) % 4 or not _B64.match(v):
        return None
    try:
        return base64.b64decode(v, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None


def clean_numeric(s: pd.Series, counts: dict | None = None) -> pd.Series:
    """'{"v": 59}' -> 59, '51-56' -> 53.5, '118274 USD'/'$118274' -> 118274, 'Val_52' -> 52, tokens -> NaN."""
    c = counts if counts is not None else {}
    out = np.full(len(s), np.nan)
    for i, raw in enumerate(s.astype(object)):
        v = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw).strip()
        if v.lower() in MISSING_TOKENS:
            c["missing"] = c.get("missing", 0) + 1
            continue
        if v.startswith("{"):
            v = _unwrap_json(v)
            c["json"] = c.get("json", 0) + 1
        if v.lower().startswith("val_"):
            v = v[4:]
            c["prefix"] = c.get("prefix", 0) + 1
        if v.upper().endswith("USD") or v.startswith("$"):
            v = v.replace("USD", "").replace("$", "").strip()
            c["currency"] = c.get("currency", 0) + 1
        m = _RANGE.match(v)
        if m:
            out[i] = (float(m[1]) + float(m[2])) / 2
            c["range"] = c.get("range", 0) + 1
            continue
        try:
            out[i] = float(v)
        except ValueError:
            c["unknown"] = c.get("unknown", 0) + 1
    return pd.Series(out, index=s.index, dtype=float)


def clean_categorical(s: pd.Series, vocab: list[str], counts: dict | None = None) -> pd.Series:
    """Decode base64 and reversed strings back onto the known vocabulary."""
    c = counts if counts is not None else {}
    lookup = {v.lower(): v for v in vocab}
    rev = {v[::-1].lower(): v for v in vocab}
    out = []
    for raw in s.astype(object):
        v = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw).strip()
        if v.lower() in MISSING_TOKENS:
            c["missing"] = c.get("missing", 0) + 1
            out.append(np.nan)
            continue
        if v.lower() in lookup:
            out.append(lookup[v.lower()])
            continue
        if v.lower() in rev:
            c["reversed"] = c.get("reversed", 0) + 1
            out.append(rev[v.lower()])
            continue
        dec = _try_b64(v)
        if dec is not None and dec.strip().lower() in lookup:
            c["base64"] = c.get("base64", 0) + 1
            out.append(lookup[dec.strip().lower()])
            continue
        c["unknown"] = c.get("unknown", 0) + 1
        out.append(np.nan)
    return pd.Series(out, index=s.index, dtype=object)


def clean_id(s: pd.Series, counts: dict | None = None) -> pd.Series:
    c = counts if counts is not None else {}
    def parse(v):
        v = str(v).strip()
        if v.lower().startswith("0x"):
            c["hex"] = c.get("hex", 0) + 1
            return int(v, 16)
        return int(float(v))
    return s.map(parse).astype(int)


def clean(raw: pd.DataFrame, drop_bad_target: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean_df, report). report is indexed by column with one count per fix."""
    df = raw.copy()
    report = {}
    if "ID" in df:
        report["ID"] = c = {}
        df["ID"] = clean_id(df["ID"], c)
    for col in NUMERIC_COLS:
        if col in df:
            report[col] = c = {}
            df[col] = clean_numeric(df[col], c)
    for col, vocab in CATEGORICAL_VOCAB.items():
        if col in df:
            report[col] = c = {}
            df[col] = clean_categorical(df[col], vocab, c)
    if drop_bad_target and TARGET in df:
        df = df[df[TARGET].notna()].reset_index(drop=True)
    rep = pd.DataFrame(report).T.reindex(columns=REPORT_COLS).fillna(0).astype(int)
    return df, rep


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ps3.csv"
    df, rep = clean(pd.read_csv(src, dtype=str, keep_default_na=False))
    print(rep.to_string())
    print(f"\nrows kept: {len(df)}")
    df.to_csv("data/processed.csv", index=False)
