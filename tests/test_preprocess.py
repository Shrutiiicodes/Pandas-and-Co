import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    clean, clean_categorical, clean_numeric, clean_id, CATEGORICAL_VOCAB,
)


def test_numeric_handles_every_corruption():
    s = pd.Series(["59", '{"v": 59}', "51-56", "118274 USD", "nan", "NULL",
                   "MISSING", "?", "", "7.3-12.3", None, "Val_52", "$67508", "Val_81.92"])
    out = clean_numeric(s)
    exp = [59, 59, 53.5, 118274, np.nan, np.nan, np.nan, np.nan, np.nan, 9.8, np.nan, 52, 67508, 81.92]
    assert out.dtype.kind == "f"
    np.testing.assert_allclose(out.to_numpy(), exp, equal_nan=True)


def test_categorical_decodes_base64_and_reversal():
    vocab = CATEGORICAL_VOCAB["Industry"]
    s = pd.Series(["Retail", "UmV0YWls", "liateR", "?", "", "nan", "Finance"])
    out = clean_categorical(s, vocab)
    assert out.tolist()[:3] == ["Retail", "Retail", "Retail"]
    assert out.isna().tolist() == [False, False, False, True, True, True, False]


def test_categorical_unknown_value_becomes_nan():
    out = clean_categorical(pd.Series(["Plumbing"]), CATEGORICAL_VOCAB["Industry"])
    assert out.isna().all()


def test_id_parses_hex_and_int():
    assert clean_id(pd.Series(["0x1", "2", "0x10"])).tolist() == [1, 2, 16]


def test_clean_end_to_end_reports_and_drops_bad_labels():
    raw = pd.DataFrame({
        "ID": ["0x1", "2", "3"],
        "Age": ['{"v": 25}', "30-34", "?"],
        "Education_Level": ["Diploma", "UGhE", "rolehcaB"],
        "Years_of_Experience": ["1.5", "nan", "2-4"],
        "Current_or_Target_Role": ["Data Analyst", "tsylanA ataD", "NULL"],
        "Technical_Skill_Score": ["50", "60", "70"],
        "Soft_Skill_Score": ["50", "60", "70"],
        "Domain_Knowledge_Score": ["50", "60", "70"],
        "Certification_Count": ["1", "2", "3"],
        "Training_Hours_Last_Year": ["10", "20", "30"],
        "Skill_Gap_Score": ["10", "20", "30"],
        "Readiness_Score": ["10", "20", "30"],
        "Employment_Status": ["Employed", "U3R1ZGVudA==", "deyolpmenU"],
        "Industry": ["IT", "UmV0YWls", "MISSING"],
        "Annual_Salary_USD": ["1000 USD", "2000", "?"],
        "Career_Readiness_Level": ["High", "TWVkaXVt", "MISSING"],
    })
    df, report = clean(raw)
    assert len(df) == 2                          # row 3 label unrecoverable
    assert df["Career_Readiness_Level"].tolist() == ["High", "Medium"]
    assert df["Education_Level"].tolist() == ["Diploma", "PhD"]
    assert df["Age"].tolist() == [25.0, 32.0]
    assert report.loc["Age", "json"] == 1
    assert report.loc["Education_Level", "base64"] == 1
    assert report.loc["Education_Level", "reversed"] == 1
    assert report.loc["Career_Readiness_Level", "missing"] == 1
