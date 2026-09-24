# Pandas & Co. · Career Readiness · PS3

---
## 1. The problem
50,000 career profiles. Predict Career_Readiness_Level (Low / Medium / High).
Catch: the file is corrupted ten different ways in every column.

---
## 2. The mess
`nan` `NULL` `MISSING` `?` · `{"v": 59}` · `51-56` · `Val_52` · `$67508` · `118274 USD`
`UmV0YWls` (base64 → Retail) · `tsylanA ataD` (reversed → Data Analyst) · `0x1`
Even the label: `muideM`, `TWVkaXVt`.
→ reports/01_corruption_heatmap.png

---
## 3. The pipeline
One small pure function per corruption. `clean(df)` composes them and counts every fix.
Result: 0 unparseable values left. 39,946 labelled rows kept.
Same function in training and in the app: no train/serve mismatch. 5 tests.
→ reports/02_recovery.png

---
## 4. We tried to predict. Honestly.
Logistic regression, gradient boosting, with and without Readiness_Score.
All at 0.334–0.337. Chance is 0.336. Shuffled labels: 0.334.
Chi-square on every categorical: p > 0.36. All feature correlations < 0.02.
→ reports/07_model_vs_chance.png

---
## 5. The finding
The label is a fair three-sided coin. The features are independent uniform noise.
Nobody can beat 33%. Anyone who does is leaking.
We report that instead of hiding it.

---
## 6. What we ship instead
Transparent Readiness Index: 40% skills, 20% experience, 20% closed gap, 10% certs, 10% training.
Explainable, tunable, no pretend learning.
Live demo: upload any dirty CSV → cleaned → audited → indexed.

---
## 7. Takeaways
Clean first, rigorously. Test for leakage. Test against chance.
A truthful "no signal" beats a fake 99%.
