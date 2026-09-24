"""Generate the story charts into reports/. Run after src.preprocess has produced data/processed.csv."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocess import CATEGORICAL_VOCAB, TARGET, clean

sns.set_theme(style="whitegrid", palette="deep")
raw = pd.read_csv("data/raw/ps3.csv", dtype=str, keep_default_na=False)
df, report = clean(raw)

# 1. corruption heatmap
rep = report.drop(index="ID").drop(columns=["unknown", "hex"])
plt.figure(figsize=(10, 6))
sns.heatmap(rep / len(raw) * 100, annot=True, fmt=".0f", cmap="rocket_r", cbar_kws={"label": "% of rows"})
plt.title("Corruption catalogue: % of rows affected per column and type")
plt.tight_layout(); plt.savefig("reports/01_corruption_heatmap.png", dpi=150); plt.close()

# 2. recovery: raw usable vs recovered vs missing
usable_raw = (raw.drop(columns="ID").apply(lambda s: pd.to_numeric(s, errors="coerce").notna()
              | s.isin(sum(CATEGORICAL_VOCAB.values(), []))).mean() * 100)
recovered = (df.reindex(columns=usable_raw.index).notna().mean() * 100)
comp = pd.DataFrame({"Usable as-is": usable_raw, "Usable after cleaning": recovered})
comp.plot.barh(figsize=(9, 6)); plt.xlabel("% of rows"); plt.title("What cleaning recovers (rows with a label kept)")
plt.tight_layout(); plt.savefig("reports/02_recovery.png", dpi=150); plt.close()

# 3. class balance
plt.figure(figsize=(5, 4))
df[TARGET].value_counts().reindex(["Low", "Medium", "High"]).plot.bar(color=["#c44e52", "#dd8452", "#55a868"])
plt.title("Career_Readiness_Level is perfectly balanced"); plt.ylabel("rows")
plt.tight_layout(); plt.savefig("reports/03_class_balance.png", dpi=150); plt.close()

# 4. numeric distributions after cleaning
num = df.select_dtypes("number").drop(columns="ID")
num.hist(figsize=(12, 8), bins=30, color="#4c72b0"); plt.suptitle("Cleaned numeric distributions (all uniform)")
plt.tight_layout(); plt.savefig("reports/04_distributions.png", dpi=150); plt.close()

# 5. correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-0.2, vmax=0.2)
plt.title("Feature correlations: nothing above |0.02|")
plt.tight_layout(); plt.savefig("reports/05_correlation.png", dpi=150); plt.close()

# 6. features by label
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, c in zip(axes.flat, ["Readiness_Score", "Skill_Gap_Score", "Technical_Skill_Score",
                             "Years_of_Experience", "Annual_Salary_USD", "Training_Hours_Last_Year"]):
    sns.boxplot(data=df, x=TARGET, y=c, order=["Low", "Medium", "High"], ax=ax)
    ax.set_xlabel("")
fig.suptitle("Every feature looks identical across readiness levels")
plt.tight_layout(); plt.savefig("reports/06_features_by_label.png", dpi=150); plt.close()

# 7. model vs chance vs shuffled
try:
    m = json.load(open("reports/metrics.json"))
    rows = {"Chance": m["chance_accuracy"], "Shuffled label": m["permutation_test"]["mean"]}
    rows |= {k.replace("/", "\n"): v["cv_accuracy"] for k, v in m["models"].items()}
    plt.figure(figsize=(10, 4.5))
    ax = pd.Series(rows).plot.bar(color=["grey", "grey"] + ["#4c72b0"] * 4)
    ax.set_ylim(0.30, 0.36); ax.set_ylabel("5-fold CV accuracy"); plt.xticks(rotation=0)
    plt.title("No model beats chance: the label carries no signal")
    plt.tight_layout(); plt.savefig("reports/07_model_vs_chance.png", dpi=150); plt.close()
except FileNotFoundError:
    print("metrics.json missing, skipped chart 7")
print("charts written to reports/")
