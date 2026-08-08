import os
import joblib
import streamlit as st
import pandas as pd
import shap
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(page_title="Fee Defaulter Predictor", page_icon="🎓", layout="centered")

st.markdown("""
<style>
    .main { background-color: #fafafa; }
    .risk-card {
        padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
        color: white; text-align: center;
    }
    .risk-high { background: linear-gradient(135deg, #e63946, #d62828); }
    .risk-moderate { background: linear-gradient(135deg, #f4a261, #e76f51); }
    .risk-low { background: linear-gradient(135deg, #2a9d8f, #264653); }
    .risk-card h1 { font-size: 2.8rem; margin: 0; }
    .risk-card p { font-size: 1rem; opacity: 0.9; margin: 0.3rem 0 0 0; }
    .factor-box {
        background: white; border-radius: 8px; padding: 0.8rem 1rem;
        margin: 0.4rem 0; border-left: 4px solid #ccc;
    }
    .factor-risk { border-left-color: #e63946; }
    .factor-protective { border-left-color: #2a9d8f; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    threshold = joblib.load("threshold.pkl")
    defaults = joblib.load("defaults.pkl")
    explainer = shap.TreeExplainer(model)
    return model, feature_cols, threshold, defaults, explainer

model, feature_cols, threshold, defaults, explainer = load_artifacts()

# ── Codebook (for readable dropdowns and display) ─────────────
MARITAL_STATUS = {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto union", 6: "Legally separated"}
GENDER = {1: "Male", 0: "Female"}
YES_NO = {1: "Yes", 0: "No"}
COURSE = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education", 9991: "Management (evening)",
}

# ── Helpers ─────────────────────────────────────────────────────
def prettify(name):
    if "_" in name and name.islower():
        return name.replace("_", " ").capitalize()
    return name

def decode_value(name, value):
    """Translate coded categorical values into their real meaning before
    they're shown to a human or sent to the LLM — prevents both misleading
    display and LLM hallucination about what a bare number represents."""
    if name == "Course":
        return COURSE.get(int(value), f"Unknown course ({value})")
    if name == "Marital status":
        return MARITAL_STATUS.get(int(value), value)
    if name == "Gender":
        return GENDER.get(int(value), value)
    if name in ["Debtor", "Scholarship holder", "Displaced", "Educational special needs", "International"]:
        return YES_NO.get(int(value), value)
    return value

def explain_prediction(row_df, top_n=3):
    shap_values = explainer.shap_values(row_df)
    if isinstance(shap_values, list):
        sv_row = shap_values[1][0]
    elif shap_values.ndim == 3:
        sv_row = shap_values[0, :, 1]
    else:
        sv_row = shap_values[0]

    ranked = sorted(zip(feature_cols, sv_row), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    result = []
    for name, shap_val in ranked:
        actual_value = row_df.iloc[0][name]
        direction = "increases risk" if shap_val > 0 else "decreases risk"
        result.append((name, actual_value, direction))
    return result

def template_advisory(risk_score, factors):
    band = "high" if risk_score >= 0.6 else "moderate" if risk_score >= 0.35 else "low"
    risk_f = [prettify(n) for n, v, d in factors if d == "increases risk"]
    prot_f = [prettify(n) for n, v, d in factors if d == "decreases risk"]
    text = f"This student shows a {band} default risk ({risk_score:.0%}). "
    if risk_f:
        text += f"Risk factors: {', '.join(risk_f)}. "
    if prot_f:
        text += f"Protective factors: {', '.join(prot_f)}. "
    text += "Suggested action: reach out before the next payment deadline to discuss a structured installment plan."
    return text

def generate_advisory(risk_score, factors):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return template_advisory(risk_score, factors) + "\n\n*(No GROQ_API_KEY set — rule-based fallback)*"
    try:
        client = Groq(api_key=api_key)
        factor_text = "; ".join(f"{prettify(n)} = {decode_value(n, v)} ({d})" for n, v, d in factors)
        prompt = (
            f"A student has a fee-default risk score of {risk_score:.0%}. "
            f"A machine learning model identified these as the top contributing factors, "
            f"with the student's actual values: {factor_text}. "
            f"Note: grades in this dataset are on a 0-20 scale (not a 4.0 GPA), "
            f"and approval_rate is a fraction between 0 and 1. "
            f"These are statistical associations the model found, not established causal "
            f"or psychological explanations — do not speculate about why a factor affects "
            f"risk (for example, do not claim a course choice 'indicates engagement' or "
            f"invent a behavioral reason). State only what the values are and what action "
            f"to take. Write a 2-3 sentence recommendation for a school financial admin — "
            f"specific and actionable, grounded only in the values given."
        )
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return template_advisory(risk_score, factors) + f"\n\n*(LLM call failed: {e} — rule-based fallback)*"

# ── UI ────────────────────────────────────────────────────────
st.title("🎓 Fee Defaulter Predictor")
st.caption("Predicts a student's risk of falling behind on tuition, with an AI-generated recommendation.")

with st.sidebar:
    st.subheader("About")
    st.write("Trained on the UCI 'Predict Students' Dropout and Academic Success' dataset (Realinho et al., 2021).")
    st.write(f"Deployed decision threshold: **{threshold:.3f}** (tuned for recall on the defaulter class).")

with st.form("student_form"):
    st.subheader("Student Details")
    col1, col2 = st.columns(2)

    with col1:
        marital_status = st.selectbox("Marital Status", list(MARITAL_STATUS), format_func=lambda x: MARITAL_STATUS[x])
        gender = st.selectbox("Gender", list(GENDER), format_func=lambda x: GENDER[x])
        course = st.selectbox("Course", list(COURSE), format_func=lambda x: COURSE[x])
        age = st.number_input("Age at Enrollment", min_value=16, max_value=70, value=20)
        scholarship = st.selectbox("Scholarship Holder", [0, 1], format_func=lambda x: YES_NO[x])
        debtor = st.selectbox("Currently a Debtor", [0, 1], format_func=lambda x: YES_NO[x])

    with col2:
        admission_grade = st.slider("Admission Grade (0–200)", 0.0, 200.0, 126.1)
        cu1_enrolled = st.number_input("Sem 1: Units Enrolled", min_value=0, value=6)
        cu1_approved = st.number_input("Sem 1: Units Approved", min_value=0, value=5)
        cu1_grade = st.slider("Sem 1: Avg Grade (0–20)", 0.0, 20.0, 12.3)
        cu2_enrolled = st.number_input("Sem 2: Units Enrolled", min_value=0, value=6)
        cu2_approved = st.number_input("Sem 2: Units Approved", min_value=0, value=5)
        cu2_grade = st.slider("Sem 2: Avg Grade (0–20)", 0.0, 20.0, 12.2)

    submitted = st.form_submit_button("Predict Risk", use_container_width=True)

if submitted:
    row = defaults.copy()
    row.update({
        "Marital status": marital_status, "Gender": gender, "Course": course,
        "Age at enrollment": age, "Scholarship holder": scholarship, "Debtor": debtor,
        "Admission grade": admission_grade,
        "Curricular units 1st sem (enrolled)": cu1_enrolled,
        "Curricular units 1st sem (approved)": cu1_approved,
        "Curricular units 1st sem (grade)": cu1_grade,
        "Curricular units 1st sem (evaluations)": max(cu1_enrolled, 1),
        "Curricular units 2nd sem (enrolled)": cu2_enrolled,
        "Curricular units 2nd sem (approved)": cu2_approved,
        "Curricular units 2nd sem (grade)": cu2_grade,
        "Curricular units 2nd sem (evaluations)": max(cu2_enrolled, 1),
    })
    row["approval_rate_1st_sem"] = round(cu1_approved / cu1_enrolled, 3) if cu1_enrolled else 0
    row["approval_rate_2nd_sem"] = round(cu2_approved / cu2_enrolled, 3) if cu2_enrolled else 0
    row["zero_units_1st_sem"] = int(cu1_enrolled == 0)
    row["zero_units_2nd_sem"] = int(cu2_enrolled == 0)
    row["grade_trend"] = round(cu2_grade - cu1_grade, 3)

    row_df = pd.DataFrame([row])[feature_cols]
    risk_score = model.predict_proba(row_df)[0][1]
    factors = explain_prediction(row_df)

    st.divider()

    band_class = "risk-high" if risk_score >= 0.6 else "risk-moderate" if risk_score >= threshold else "risk-low"
    band_label = "HIGH RISK" if risk_score >= 0.6 else "MODERATE RISK" if risk_score >= threshold else "LOW RISK"

    st.markdown(f"""
    <div class="risk-card {band_class}">
        <h1>{risk_score:.0%}</h1>
        <p>{band_label} — probability of falling behind on fees</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Why this prediction?")
    for name, value, direction in factors:
        display_value = decode_value(name, value)
        css_class = "factor-risk" if direction == "increases risk" else "factor-protective"
        icon = "🔺" if direction == "increases risk" else "🔻"
        st.markdown(f"""
        <div class="factor-box {css_class}">
            {icon} <b>{prettify(name)}</b> = {display_value} — {direction}
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🤖 AI-Generated Recommendation")
    with st.spinner("Generating advisory..."):
        advisory = generate_advisory(risk_score, factors)
    st.info(advisory)

st.divider()
st.caption("Built by Muiz Adeyemi — Flexisaf Generative AI & Data Science Capstone.")