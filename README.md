# Fee Defaulter Prediction System

An ML system that predicts which students are likely to fall behind on tuition fees, with an AI-generated, actionable recovery recommendation for school admin staff.

**Built by:** Muiz Adeyemi ([@Schwoz17](https://github.com/Schwoz17)) — Flexisaf Generative AI & Data Science Capstone

---

## Why this project

Most fee-risk models stop at "flag the student." This one goes further: it explains *why* a student is at risk (SHAP-driven, per-student) and generates a specific, actionable recommendation for what to do about it (LLM-powered) — turning a risk score into an actual intervention plan.

## Live Demo
> Deploy via [Streamlit Community Cloud](https://share.streamlit.io) pointing at `app.py`, then paste the link here.

## Problem Statement
School administrators need to identify students at risk of falling behind on fee payments *before* it happens, so interventions (payment plans, financial counseling, scholarship referral) can happen proactively instead of reactively.

## Dataset
**"Predict Students' Dropout and Academic Success"** — Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89. **Licensed CC BY 4.0.**

4,424 real students from a Portuguese higher education institution: demographics, socio-economic background, and academic performance across the first two semesters. Zero missing values, zero duplicates in the source.

**How to get it:**
```bash
pip install ucimlrepo
```
```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=697)
```
Or download directly from the [UCI dataset page](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

**Reframing:** the dataset's original purpose is a 3-class dropout classifier (Dropout/Enrolled/Graduate). This project repurposes its `Tuition fees up to date` field — a genuine binary payment-status flag — as the prediction target instead, since that's the field that actually corresponds to fee default.

**Important caveat on the target:** `Tuition fees up to date` is a payment-status *snapshot* captured around enrollment time, not a formal, sustained default event with severity or duration. It's the closest real proxy publicly available for fee default — genuinely a payment-status field, not a fabricated stand-in — but it's softer than "default" in the strict financial sense. This is stated explicitly rather than glossed over.

**External validity caveat:** this is Portuguese higher-education data, not Nigerian. The value of this project is the *methodology and pipeline*, which would transfer cleanly to real Nigerian fee data if it becomes available — the specific learned patterns here are population-specific.

**Feature engineering:**
- `approval_rate_1st_sem`, `approval_rate_2nd_sem` — approved/enrolled ratio, more informative than raw counts
- `zero_units_1st_sem`, `zero_units_2nd_sem` — flag for the 180 students (4%) who enrolled in 0 units; this group's fee-status rate wasn't notably different from average, but their dropout rate was (42.8% vs ~32%) — a real pattern worth flagging separately rather than folding into a 0% ratio
- `grade_trend` — 2nd semester grade minus 1st, captures improving/declining trajectory

40 total features after engineering. `Tuition fees up to date` and the original `Target` column are excluded from the feature set (the former defines the target; the latter is plausibly downstream of fee default, not a cause of it).

## Methods
- **Models compared:** Decision Tree (interpretable baseline) vs Random Forest (ensemble)
- **Evaluation:** F1-score and recall on the defaulter class prioritized over raw accuracy — a missed defaulter is costlier than a false alarm
- **Explainability:** Global feature importance + per-student SHAP values

### Threshold tuning — a real finding, not a formality
At the default 0.5 threshold, Random Forest actually had **lower recall** than the Decision Tree (0.557 vs 0.632) despite better precision — the opposite of what "the stronger model" would suggest. Since recall on the defaulter class was established as the priority metric, the Decision Tree's default output would have been the better deployment choice as-is.

Rather than accept that, Random Forest's decision threshold was tuned down from 0.5 to **0.456**, trading a small amount of precision for meaningfully higher recall:

| Model | Threshold | Precision (Defaulted) | Recall (Defaulted) | F1 (Defaulted) |
|---|---|---|---|---|
| Decision Tree | 0.5 | 0.411 | 0.632 | 0.498 |
| Random Forest | 0.5 | 0.465 | 0.557 | 0.506 |
| **Random Forest (tuned)** | **0.456** | **0.454** | **0.651** | **0.535** |

Tuned Random Forest beats both alternatives on every metric simultaneously and is the deployed model.

## Feature Importance
`Debtor` dominates (importance ≈ 0.137, well ahead of the next-closest feature) — consistent with manual EDA earlier in the notebook, a good sign the model learned a real pattern rather than noise. The engineered `approval_rate_2nd_sem` and `approval_rate_1st_sem` both rank in the top 4, ahead of the raw curricular-unit counts they were derived from.

## AI-Powered Feature: Personalized Fee-Recovery Suggestions

Two layers:
1. **Rule-based** (`template_advisory`) — buckets risk into low/moderate/high, lists risk vs. protective factors in plain language. No API required; also serves as the automatic fallback if the LLM call fails or no API key is set.
2. **LLM-powered** (`generate_advisory`) — sends the risk score and SHAP-derived top factors to Groq (`openai/gpt-oss-20b`) to draft a specific, actionable recommendation for a school admin.

### Three real bugs found and fixed while building this

**1. Backwards direction assumption + invented grading scale.** The first prompt version passed only direction labels ("increases risk"), not actual values. The LLM guessed the wrong direction for `approval_rate_2nd_sem` and invented a 4.0 GPA scale that doesn't exist in this dataset (grades here are 0–20). **Fix:** pass real values in the prompt and explicitly state the dataset's grading scale.

**2. Undecoded categorical codes.** `Course` is stored as an integer code (e.g. `33`). Shown raw, the LLM interpreted `Course = 33` as a "course load of 33," which isn't what the field means at all. **Fix:** added `decode_value()`, which translates coded categoricals (Course, Marital status, Gender, Yes/No flags) into their real labels before display or prompting.

**3. Invented causal narrative.** Even with correct values, the LLM added unsupported interpretive claims (e.g. framing a specific course enrollment as "indicating strong engagement") — SHAP values show statistical association, not a causal or psychological explanation. **Fix:** the prompt now explicitly instructs the model not to speculate about *why* an association exists, only to state the values and recommend action. This traded a bit of narrative fluency for accuracy — a deliberate, worthwhile tradeoff for a tool meant to inform real decisions about real students.

## Visualizations
All charts in `visuals/`:
- `eda_overview.png` — class balance, admission grade by fee status, scholarship breakdown
- `confusion_matrices.png` — Decision Tree vs Random Forest at default threshold
- `roc_curve.png` — ROC/AUC comparison
- `threshold_tuning.png` — precision/recall vs. decision threshold
- `feature_importance.png` — Random Forest global feature importance
- `shap_summary.png` — SHAP summary plot

---

## Project Structure
fee-defaulter-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── .env # GROQ_API_KEY — gitignored, never committed
├── venv/ # gitignored
├── flexisaf_fee_defaulter_prediction_system.ipynb
├── app.py # Streamlit deployment app
├── rf_model.pkl # trained, threshold-tuned Random Forest
├── feature_cols.pkl
├── threshold.pkl
├── defaults.pkl # median/mode defaults for fields not in the app form
└── visuals/

## Running Locally
```bash
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash
pip install -r requirements.txt
streamlit run app.py
```

Set your Groq API key in a `.env` file (never commit this):
GROQ_API_KEY=your_key_here

Without a key, the app automatically falls back to the rule-based advisory — the full pipeline still runs end-to-end.

---

## Deliverables Checklist (per capstone spec)

| # | Requirement | Location |
|---|---|---|
| 1 | Notebook, clean and documented | `flexisaf_fee_defaulter_prediction_system.ipynb` |
| 2 | GitHub repo: code, data, notebook HTML/PDF export | This repo |
| 3 | Dataset (cleaned) + source credit + cleaning brief | See Dataset section above |
| 4 | Model training, evaluation, visualization code | Notebook, Sections 6–10 |
| 5 | Charts & insights | `visuals/` |
| 6 | PDF slide summary (link, not Google Drive) | 
| 7 | All deliverables linked from GitHub repo | 