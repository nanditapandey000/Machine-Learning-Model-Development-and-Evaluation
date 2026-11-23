# Machine-Learning-Model-Development-and-Evaluation

# Week 4 — Machine Learning Model Development

**Project:** Predicting LendingClub Loan Defaults
**Focus:** Supervised classification pipeline, model comparison, feature selection
**Dataset:** LendingClub LoanStats3a (42,535 loans, 2007–2011)
**Report date:** November 2026

---

## 1. Objective

Week 4 builds the predictive-modeling layer of the project. Given the binary target
`is_default`, the task is to train, tune, and compare several classifiers, select a
champion by cross-validated performance, and explain *why* the champion wins. The
deliverable is a reproducible ML pipeline plus a narrative report that documents
feature engineering, hyperparameter search, and the trade-offs between model families.

---

## 2. Deliverables

| File | Location | Purpose |
|------|----------|---------|
| **Week4_ML_Model_Development_Report.pdf** | `download/` | Main narrative report (cover + body) |
| **week4_ml_model_development.ipynb** | `download/` | Reproducible ML notebook |
| 6 evaluation figures | `data/week4_figures/` | ROC/PR comparison, metric bars, confusion matrices, feature importance, learning curve |
| `week4_stats.json` | `data/` | All model metrics, best params, confusion matrices |
| `best_model.joblib` | `data/` | Serialized champion model for inference |
| `week4_ml_pipeline.py` | `scripts/` | Full sklearn pipeline (preprocess → tune → evaluate) |
| `week4_feature_importance.py` | `scripts/` | Permutation-importance computation |
| `build_week4_body.py` | `scripts/` | ReportLab body PDF generator |
| `week4_cover.html` / `.pdf` | `scripts/` | Cover page |

---

## 3. Dataset & Split

- **Rows:** 42,535 loans (same Week-1 cleaned dataset).
- **Target:** `is_default` — 15.12% positive rate (6,431 defaults).
- **Features after preprocessing:** 47 (18 numeric + 4 one-hot encoded categoricals).
- **Train / test split:** 29,774 / 12,761 (70 / 30), stratified on the target.
- **Cross-validation:** 5-fold stratified, optimizing PR-AUC (the right metric at
  this imbalance level — ROC-AUC overstates performance on rare-positive problems).

---

## 4. Models Compared

| Model | ROC-AUC | PR-AUC | F1 | Notes |
|-------|---------|--------|-----|-------|
| **Logistic Regression** (champion) | 0.697 | **0.288** | 0.354 | Tuned C ≈ 0.98, L2 penalty |
| Random Forest | 0.692 | 0.276 | 0.331 | 200 trees, max_depth tuned |
| Gradient Boosting | 0.689 | 0.271 | 0.318 | 150 estimators, lr = 0.05 |
| Dummy (stratified) | 0.500 | 0.151 | 0.041 | Baseline |

**Champion: Logistic Regression.** The simplest model wins on PR-AUC, the metric we
optimized for. This is the central finding of the week — and it is consistent with
the Week-3 statistical analysis: the dominant predictive feature (loan grade) is
monotonic in the target, and linear models capture monotonic relationships
efficiently. The tree ensembles' extra capacity buys very little lift because there
are few strong non-linearities left after grade is accounted for.

---

## 5. Pipeline Architecture

```
raw features
  │
  ├─ numeric: median impute → standard scale
  ├─ categorical: constant impute → one-hot encode (drop='first')
  │
  ├─ feature selection: drop int_rate (collinear with grade, Week-3 finding)
  │
  ├─ model: LogisticRegression / RandomForest / GradientBoosting
  │
  ├─ hyperparameter search: Optuna, 30 trials, 5-fold stratified CV
  │                         objective = mean PR-AUC
  │
  └─ evaluate on held-out test set: ROC-AUC, PR-AUC, F1, confusion matrix
```

The pipeline is a single `sklearn.pipeline.Pipeline` object, so the same
preprocessing is applied identically to train and test — no leakage.

---

## 6. Key Figures

| File | Content |
|------|---------|
| `m01_roc_comparison.png` | ROC curves for all four models on one axis |
| `m02_pr_comparison.png` | Precision-recall curves — the more honest comparison at 15% base rate |
| `m03_metric_bars.png` | Side-by-side bar chart of ROC-AUC, PR-AUC, F1 |
| `m04_confusion_matrices.png` | 2×2 grid of confusion matrices (one per model) |
| `m05_feature_importance.png` | Permutation importance of the top 15 features in the champion model |
| `m06_learning_curve.png` | Learning curve — is the champion underfitting or overfitting? |

---

## 7. Feature Importance (Champion Model)

Permutation-importance top 5 (mean PR-AUC drop when shuffled):

1. **grade_ord** — by a wide margin, the most important feature.
2. **int_rate** (kept as a sanity check despite the collinearity note — see §8).
3. **inq_last_6mths** — recent credit inquiries, a behavioral signal.
4. **loan_to_income** — engineered ratio (loan amount ÷ annual income).
5. **revol_util** — revolving credit utilization.

The ranking mirrors the Week-3 logistic-regression coefficient ordering, which is
the expected cross-validation: two different methods, same conclusion.

---

## 8. Honest Limitations

- **Champion performance is modest.** PR-AUC of 0.288 means precision-recall is
  genuinely hard on this problem — the dataset's signal ceiling is around 0.30.
  No model family in our search breaks through that ceiling.
- **Class imbalance is real.** With 15% positives, we tuned for PR-AUC rather than
  ROC-AUC, but did not apply SMOTE or class-weighting. Adding class weights would
  likely trade precision for recall; we judged the default-threshold F1 = 0.35 to be
  the more operationally useful operating point.
- **`int_rate` was kept** despite the Week-3 collinearity warning. We verified that
  removing it dropped PR-AUC by < 0.005, so the leak through grade is not severe
  here. A stricter pipeline would drop it; we document the trade-off rather than hide it.
- **Time split vs random split.** We used a stratified random split. A more rigorous
  evaluation would split by origination date (train on 2007–2010, test on 2011) to
  simulate real deployment. We did not do this; it is flagged for future work.

---

## 9. How to Reproduce

```bash
# From the repository root
python3 -I scripts/week4_ml_pipeline.py          # trains, tunes, evaluates, saves best_model.joblib
python3 -I scripts/week4_feature_importance.py   # computes permutation importance
python3 -I scripts/build_week4_body.py           # builds scripts/body_week4.pdf
python3 skills/pdf/scripts/pdf.py code.sanitize scripts/build_week4_body.py
```

**Notebook:** upload `download/cleaned_dataset.csv` and Run All. The notebook runs
the full pipeline end-to-end and reproduces every figure and metric.

---

## 10. visualization portfolio:

- The dominance of `grade_ord` → Figure 3 (grade ladder) and Figure 8 (correlation
  heatmap).
- The interaction signal that tree ensembles *should* have exploited but didn't
  decisively → Figure 10 (purpose × grade risk matrix) shows the interaction exists
  but is small relative to the main effect.
- The modest predictive ceiling → Figures 6, 7, 9 explain *why*: single continuous
  features (interest rate, DTI, loan-to-income) overlap heavily between paid and
  defaulted loans.

Week 5 thus closes the analytical loop: the patterns the models found, the
visualizations explain.
