# app.py — Full Health Assistant v3 (designer UI + DB migration)
# Place this next to best_model.pkl
# Optional: pip install streamlit joblib pandas numpy plotly matplotlib fpdf shap

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import io
import os
import time
from datetime import datetime

# Optional libraries
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

import plotly.graph_objects as go
import matplotlib.pyplot as plt

# -------------------- Styling / Theme --------------------
st.set_page_config(page_title="Health Assistant", page_icon="🧠", layout="wide")

def set_bg_hack_url():
    '''
    A function to set a dimmed background image using linear gradient overlay.
    '''
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)),
                        url(https://i.fbcd.co/products/resized/resized-750-500/a-medical-illustration-background-for-worlds-health-day-with-copy-space-4-cf3c72b8b3657425f48024f3055903e4f8ac17cd3cea2666b4277f9448c846ae.jpg);
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Custom Teal Glow Tabs
st.markdown("""
<style>

div[data-testid="stTabs"] > div {
    background-color: rgba(255,255,255,0.7);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 6px 8px;
    box-shadow: 0px 4px 12px rgba(0, 180, 170, 0.1);
    border: 1px solid rgba(0,180,170,0.15);
}

div[data-testid="stTabs"] button {
    border-radius: 12px !important;
    padding: 8px 18px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #046B6B !important;
    background-color: rgba(255,255,255,0.6) !important;
    border: 1px solid rgba(0,180,170,0.15) !important;
    transition: all .3s ease-in-out !important;
}

div[data-testid="stTabs"] button:hover {
    color: #00A6A6 !important;
    border-color: #00A6A6 !important;
    background: rgba(230,255,253,0.9) !important;
    box-shadow: 0px 0px 12px rgba(0,200,190,0.35);
}

div[data-testid="stTabs"] button[data-baseweb="tab-highlight"] {
    color: white !important;
    background: linear-gradient(135deg, #00C2C2, #009B9B) !important;
    border-color: #00C2C2 !important;
    box-shadow: 0px 0px 15px rgba(0,200,190,0.45);
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)

# Call the function to set the dimmed background image
set_bg_hack_url()





# Custom CSS for nicer visuals (cards, buttons, gradients)
st.markdown(
    """
    <style>
    /* Fonts and base */
    html, body, [class*="css"]  {
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* Header */
    h1, h2, .stTabs {
        color: #4B4DED !important;
    }

    /* Card style */
    .report-card {
        background: linear-gradient(135deg, #fbfdff 0%, #eef3ff 100%);
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(80,86,255,0.08);
        border: 1px solid rgba(108,99,255,0.08);
        margin-bottom: 14px;
    }

    /* Accent boxes for results */
    .result-box {
        background: linear-gradient(90deg,#ffffff,#f6f8ff);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #e8ecff;
        box-shadow: 0 4px 14px rgba(120,126,255,0.04);
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg,#6c63ff,#8a85ff) !important;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 8px 14px;
        font-weight: 700;
    }
    div.stButton > button:hover {
        box-shadow:0 6px 22px rgba(108,99,255,0.28) !important;
    }

    /* Inputs / selects */
    input, select, textarea {
        border-radius: 8px !important;
    }

    /* Smaller captions */
    .stCaption { color: #6b7280; }

    /* breathing animation (small) */
    .breath-ball {
      width:120px;height:120px;
      background: radial-gradient(circle at 30% 30%, #7b61ff, #5b45ff);
      border-radius: 50%;
      margin: 12px auto;
      box-shadow: 0 10px 30px rgba(91,69,255,0.18);
      animation: breathe 8s infinite;
    }
    @keyframes breathe {
      0% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.35); opacity: 0.9; }
      100% { transform: scale(1); opacity: 1; }
    }

    /* Tiny info badge */
    .badge {
      display:inline-block;
      padding:6px 10px;
      border-radius:999px;
      background: linear-gradient(90deg,#ffe9d6,#ffd1d1);
      color:#7a2e2e;
      font-weight:700;
      font-size:13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top header (designer)
st.markdown("<h1 style='text-align:center;color:#2b2b8a;margin-bottom:6px;'>🧠 Health Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#565656;margin-top:0;margin-bottom:12px;'>Screening, tracking, explainability, PDF reports, stress tools, companion — for awareness only.</p>", unsafe_allow_html=True)

# -------------------- Helpers & Storage --------------------
DB_PATH = "patients.db"

def migrate_db_add_missing_columns(conn, required_cols):
    """Ensure required columns exist; add them if missing."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(patients);")
    existing = {row[1] for row in cur.fetchall()}  # row[1] is column name
    to_add = []
    for col, col_def in required_cols.items():
        if col not in existing:
            to_add.append((col, col_def))
    for col, col_def in to_add:
        # Safe ALTER TABLE to add column
        cur.execute(f"ALTER TABLE patients ADD COLUMN {col} {col_def};")
    if to_add:
        conn.commit()
    return [c for c, _ in to_add]

def init_db():
    # If DB does not exist, create table with full schema.
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        ever_married TEXT,
        hypertension INTEGER,
        heart_disease INTEGER,
        avg_glucose REAL,
        bmi REAL,
        smoking_status TEXT,
        work_type TEXT,
        residence TEXT,
        systolic_bp INTEGER,
        diastolic_bp INTEGER,
        cholesterol REAL,
        hba1c REAL,
        waist_cm REAL,
        sleep_hours REAL,
        stress_level INTEGER,
        genetic_family_history INTEGER,
        genetic_variant_count INTEGER,
        risk REAL
    )
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(create_table_sql)
    conn.commit()

    # Migration: ensure any missing columns present (in case DB was created earlier without some fields)
    required = {
        "ever_married": "TEXT",
        "avg_glucose": "REAL",
        "waist_cm": "REAL",
        "stress_level": "INTEGER",
        "genetic_family_history": "INTEGER",
        "genetic_variant_count": "INTEGER"
    }
    added = migrate_db_add_missing_columns(conn, required)
    if added:
        # small notice in logs — visible in console only
        print(f"DB migration: added columns: {added}")
    conn.close()

def save_record(rec):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Use explicit column list matching schema to avoid issues
    cur.execute("""
    INSERT INTO patients (timestamp, name, age, gender, ever_married, hypertension, heart_disease,
                          avg_glucose, bmi, smoking_status, work_type, residence, systolic_bp, diastolic_bp,
                          cholesterol, hba1c, waist_cm, sleep_hours, stress_level, genetic_family_history,
                          genetic_variant_count, risk)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        rec.get('timestamp'), rec.get('name'), rec.get('age'), rec.get('gender'), rec.get('ever_married'),
        rec.get('hypertension'), rec.get('heart_disease'), rec.get('avg_glucose'), rec.get('bmi'),
        rec.get('smoking_status'), rec.get('work_type'), rec.get('residence'), rec.get('systolic_bp'),
        rec.get('diastolic_bp'), rec.get('cholesterol'), rec.get('hba1c'), rec.get('waist_cm'),
        rec.get('sleep_hours'), rec.get('stress_level'), rec.get('genetic_family_history'),
        rec.get('genetic_variant_count'), rec.get('risk')
    ))
    conn.commit()
    conn.close()

def load_history(limit=1000):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM patients ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    except Exception:
        # If query fails (schema mismatch), attempt a safer select of available columns
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(patients);")
        cols = [r[1] for r in cur.fetchall()]
        cols_sql = ", ".join(cols)
        df = pd.read_sql_query(f"SELECT {cols_sql} FROM patients ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df

def find_pipeline_steps(pipeline):
    preproc = None
    clf = None
    if pipeline is None:
        return None, None
    if hasattr(pipeline, "named_steps"):
        names = list(pipeline.named_steps.keys())
        for p in ("preprocessor", "preproc", "prep", "transformer"):
            if p in pipeline.named_steps:
                preproc = pipeline.named_steps[p]; break
        if preproc is None:
            for name in names:
                obj = pipeline.named_steps[name]
                if obj.__class__.__name__ == "ColumnTransformer":
                    preproc = obj; break
        for c in ("classifier","clf","model","estimator","final_estimator"):
            if c in pipeline.named_steps:
                clf = pipeline.named_steps[c]; break
        if clf is None:
            last_name = names[-1]
            clf = pipeline.named_steps[last_name]
    else:
        clf = pipeline
        preproc = None
    return preproc, clf

def build_input_df(inputs):
    age = inputs["age"]
    avg_glu = inputs["avg_glucose_level"]
    bmi = inputs["bmi"]
    return pd.DataFrame([{
        "age": age,
        "gender": inputs["gender"],
        "ever_married": inputs.get("ever_married","Yes"),
        "hypertension": 1 if inputs["hypertension"] == "Yes" else 0,
        "heart_disease": 1 if inputs["heart_disease"] == "Yes" else 0,
        "work_type": inputs.get("work_type","Private"),
        "Residence_type": inputs.get("residence","Urban"),
        "avg_glucose_level": avg_glu,
        "bmi": bmi,
        "smoking_status": inputs.get("smoking_status","never smoked"),
        "age_risk": 1 if age > 60 else 0,
        "high_glucose": 1 if avg_glu > 130 else 0,
        "obesity": 1 if bmi > 30 else 0,
        "glucose_bmi_ratio": avg_glu / (bmi + 1)
    }])

def generate_pdf_bytes(name, model_name, risk_pct, status, inputs_dict, reasons, ai_advice):
    if not FPDF_AVAILABLE:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Health Risk Assessment", ln=1, align="C")
    pdf.set_font("Arial", size=11)
    pdf.ln(4)
    pdf.cell(0, 8, f"Patient: {name}", ln=1)
    pdf.cell(0, 8, f"Model: {model_name}", ln=1)
    pdf.cell(0, 8, f"Predicted Risk: {risk_pct}%", ln=1)
    pdf.cell(0, 8, f"Category: {status}", ln=1)
    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Inputs", ln=1)
    pdf.set_font("Arial", size=10)
    for k, v in inputs_dict.items():
        pdf.cell(0, 7, f"{k}: {v}", ln=1)
    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Key Drivers", ln=1)
    pdf.set_font("Arial", size=10)
    for r in reasons:
        pdf.multi_cell(0, 6, f"- {r}")
    pdf.ln(6)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Personalized AI Advice", ln=1)
    pdf.set_font("Arial", size=10)
    for a in ai_advice:
        pdf.multi_cell(0, 6, f"- {a}")
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return io.BytesIO(pdf_bytes)

# -------------------- Init DB & Model --------------------
init_db()

MODEL_PATH = "best_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}. Place pipeline (best_model.pkl) here.")
    st.stop()

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed loading pipeline: {e}")
    st.stop()

preproc, clf = find_pipeline_steps(pipeline)

# -------------------- UI Layout (tabs with emojis) --------------------
tabs = st.tabs([
    "🧮 Predict", "🔍 Explain", "📊 Dashboard", "📄 Report & Tips",
    "🌬 Stress & Breathing", "👨‍⚕️ Companion", "🧬 Genetic Risk", "ℹ️ About"
])

# -------------------- PREDICT TAB --------------------
with tabs[0]:
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Stroke Risk — Predict")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            name = st.text_input("👤 Full name (optional)")
            age = st.slider("🎚 Age", 1, 100, 45)
            gender = st.selectbox("⚧ Gender", ["Male","Female","Other"])
            ever_married = st.selectbox("💍 Ever Married?", ["Yes","No"])
        with col2:
            hypertension = st.selectbox("💢 Hypertension", ["No","Yes"])
            heart_disease = st.selectbox("❤️ Heart Disease", ["No","Yes"])
            work_type = st.selectbox("💼 Work Type", ["Private","Self-employed","Govt_job","Children","Never_worked"])
        with col3:
            residence = st.selectbox("🏡 Residence", ["Urban","Rural"])
            smoking_status = st.selectbox("🚬 Smoking Status", ["never smoked","formerly smoked","smokes","Unknown"])
            avg_glucose_level = st.number_input("🩸 Average Glucose Level (mg/dL)", 50.0, 400.0, 105.0)
        bmi = st.number_input("⚖️ BMI", 10.0, 60.0, 24.5)
        systolic_bp = st.number_input("🩺 Systolic BP (mmHg)", 80, 220, 120)
        diastolic_bp = st.number_input("🩺 Diastolic BP (mmHg)", 40, 140, 80)
        cholesterol = st.number_input("🧪 Total cholesterol (mg/dL)", 100.0, 400.0, 180.0)
        hba1c = st.number_input("🧾 HbA1c (%)", 3.0, 15.0, 5.6)
        waist_cm = st.number_input("📏 Waist circumference (cm)", 50.0, 200.0, 80.0)
        sleep_hours = st.number_input("😴 Average sleep (hours/night)", 2.0, 12.0, 7.0)
        stress_level = st.slider("⚖ Stress level (1 low — 10 high)", 1, 10, 4)
        genetic_family_history = st.checkbox("🧬 Family history of stroke/early CVD?")
        genetic_variant_count = st.number_input("🔢 Number of known high-risk variants (if available)", 0, 50, 0)
        submit = st.form_submit_button("Estimate Risk")

    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        inputs = {
            "age": age, "gender": gender, "ever_married": ever_married,
            "hypertension": hypertension, "heart_disease": heart_disease,
            "work_type": work_type, "residence": residence,
            "avg_glucose_level": avg_glucose_level, "bmi": bmi,
            "smoking_status": smoking_status
        }
        X = build_input_df(inputs)
        try:
            prob = pipeline.predict_proba(X)[0][1]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            prob = 0.0

        genetic_multiplier = 1.0
        if genetic_family_history:
            genetic_multiplier += 0.08
        genetic_multiplier += (genetic_variant_count * 0.01)
        adjusted_prob = prob * genetic_multiplier
        adjusted_prob = min(adjusted_prob, 0.999)
        risk_pct = round(adjusted_prob * 100, 2)

        if risk_pct < 20:
            ai_advice = [
                "Keep current activity; aim for 30 min/day.",
                "Monitor vitals annually and maintain healthy diet.",
                "Routine screening recommended."
            ]
        elif risk_pct < 50:
            ai_advice = [
                "Increase moderate cardio to 4–5x/week.",
                "Reduce processed carbs and monitor glucose.",
                "Check BP regularly; consult primary care if rising."
            ]
        else:
            ai_advice = [
                "Arrange clinical evaluation; consider specialist referral.",
                "Tight BP & glucose control; avoid tobacco & alcohol.",
                "Begin structured exercise program after clinical clearance."
            ]

        # Gauge (plotly)
        if risk_pct < 20:
            bar_color = "#2ecc71"
        elif risk_pct < 50:
            bar_color = "#ffa500"
        else:
            bar_color = "#ff4d4f"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={'suffix':'%'},
            title={'text':'Estimated Stroke Risk'},
            gauge={'axis':{'range':[0,100]}, 'bar':{'color':bar_color},
                   'steps':[{'range':[0,20],'color':'#e6f7e6'},{'range':[20,50],'color':'#fff7d6'},{'range':[50,100],'color':'#fdecea'}]}
        ))
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

        # Projected trend
        years = np.arange(0,6)
        growth = 1.04
        projected = np.clip(risk_pct * (growth**years), 0, 100)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=years, y=projected, mode='lines+markers'))
        fig2.update_layout(title="5-Year Projected Risk (if unchanged)", xaxis_title="Years", yaxis_title="Risk (%)")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Projection assumes no lifestyle or treatment changes.")

        reasons = []
        if age > 60: reasons.append("Age > 60")
        if hypertension == "Yes": reasons.append("Hypertension history")
        if avg_glucose_level > 130 or hba1c > 6.5: reasons.append("Elevated glucose/HbA1c")
        if bmi > 30 or waist_cm >= 102: reasons.append("Obesity/central adiposity")
        if smoking_status in ("smokes","formerly smoked"): reasons.append("Smoking exposure")
        if cholesterol > 240: reasons.append("High cholesterol")
        if systolic_bp >= 140 or diastolic_bp >= 90: reasons.append("Elevated BP")
        if stress_level >= 8: reasons.append("High stress level")
        if genetic_family_history: reasons.append("Family history of stroke/CVD")
        if genetic_variant_count > 0: reasons.append(f"{genetic_variant_count} reported high-risk genetic variants (user-provided)")
        if not reasons: reasons = ["No major concerning signals"]

        st.subheader("Why the model estimated this")
        for r in reasons:
            st.write("•", r)

        st.subheader("🧠 Smart Lifestyle Advice")
        for a in ai_advice:
            st.write("•", a)

        # Save session last input for explainability
        st.session_state['last_input'] = {
            "age": age, "gender": gender, "ever_married": ever_married,
            "hypertension": hypertension, "heart_disease": heart_disease,
            "work_type": work_type, "residence": residence,
            "avg_glucose_level": avg_glucose_level, "bmi": bmi,
            "smoking_status": smoking_status,
            "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
            "cholesterol": cholesterol, "hba1c": hba1c, "waist_cm": waist_cm,
            "sleep_hours": sleep_hours, "stress_level": stress_level,
            "genetic_family_history": 1 if genetic_family_history else 0,
            "genetic_variant_count": genetic_variant_count
        }

        # Save to DB (record uses avg_glucose key)
        record = {
            'timestamp': datetime.now().isoformat(),
            'name': name or "Anonymous",
            'age': age, 'gender': gender, 'ever_married': ever_married,
            'hypertension': 1 if hypertension == "Yes" else 0,
            'heart_disease': 1 if heart_disease == "Yes" else 0,
            'avg_glucose': avg_glucose_level, 'bmi': bmi, 'smoking_status': smoking_status,
            'work_type': work_type, 'residence': residence,
            'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp, 'cholesterol': cholesterol,
            'hba1c': hba1c, 'waist_cm': waist_cm, 'sleep_hours': sleep_hours, 'stress_level': stress_level,
            'genetic_family_history': 1 if genetic_family_history else 0,
            'genetic_variant_count': genetic_variant_count,
            'risk': risk_pct
        }
        save_record(record)

        # Downloadable reports
        st.write("---")
        st.download_button("Download Quick TXT Report", data=(
            f"Name: {name or 'Anonymous'}\nRisk: {risk_pct}%\nCategory: {'Low' if risk_pct<25 else 'Moderate' if risk_pct<50 else 'High'}\nTime: {record['timestamp']}\n"
        ), file_name=f"stroke_report_{risk_pct}.txt")

        pdf_buf = generate_pdf_bytes(name or "Anonymous", "stroke", risk_pct, "",
                                     {"age": age, "gender": gender, "glucose": avg_glucose_level, "bmi": bmi, "bp": f"{systolic_bp}/{diastolic_bp}"},
                                     reasons, ai_advice)
        if pdf_buf is not None:
            st.download_button("📄 Download PDF report", data=pdf_buf, file_name=f"StrokeReport_{risk_pct}.pdf", mime="application/pdf")
        else:
            st.info("Install `fpdf` for PDF export.")
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------- EXPLAIN TAB --------------------
with tabs[1]:
    import numpy as np

    # Fix numpy bool deprecation
    if not hasattr(np, "bool"):
        np.bool = bool

    st.header("Explainability & Local Insights")

    if 'last_input' not in st.session_state:
        st.info("No recent input. Run a prediction in Predict tab.")
    else:
        Xlast = build_input_df(st.session_state['last_input'])

        if SHAP_AVAILABLE:
            st.write("✅ SHAP available — generating explanation...")

            try:
                # Prepare data for SHAP
                if preproc is not None:
                    Xtr = preproc.transform(Xlast)
                    model_for_shap = clf if clf is not None else pipeline

                    # Get feature names safely
                    try:
                        feature_names = preproc.get_feature_names_out()
                    except:
                        feature_names = Xlast.columns
                else:
                    Xtr = Xlast
                    model_for_shap = pipeline
                    feature_names = Xlast.columns

                # Build SHAP explainer
                explainer = shap.Explainer(model_for_shap, Xtr, feature_names=feature_names)
                shap_values = explainer(Xtr)

                # Try force plot first (safer than waterfall)
                try:
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values.values,
                        Xlast,
                        matplotlib=True
                    )
                    st.pyplot(bbox_inches='tight')

                except Exception:
                    st.warning("⚠ SHAP force plot failed. Showing bar summary instead.")
                    shap.plots.bar(shap_values)
                    st.pyplot(bbox_inches='tight')

            except Exception as e:
                st.warning(f"⚠ SHAP failed: {e}")

                # Fallback to feature importance if available
                imp = getattr(clf, "feature_importances_", None)
                if imp is not None:
                    fi = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False).head(10)
                    st.bar_chart(fi.set_index("feature"))
                else:
                    st.info("No feature importance available.")
        else:
            st.info("SHAP not installed. Showing feature importance if available.")
            imp = getattr(clf, "feature_importances_", None)
            if imp is not None and preproc is not None:
                feature_names = preproc.get_feature_names_out()
                fi = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False).head(10)
                st.bar_chart(fi.set_index("feature"))
            else:
                st.write("Explainability not available in this environment.")

# -------------------- DASHBOARD TAB --------------------
with tabs[2]:
    st.header("Dashboard & History")
    hist = load_history(2000)
    if hist.empty:
        st.info("No saved records yet.")
    else:
        st.write("Recent records (newest first):")
        st.dataframe(hist)
        csv = hist.to_csv(index=False).encode('utf-8')
        st.download_button("Download history CSV", data=csv, file_name="history.csv", mime="text/csv")

        hist['ts'] = pd.to_datetime(hist['timestamp'])
        series = hist.groupby(hist.ts.dt.date)['risk'].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index.astype(str), y=series.values, mode='lines+markers', name='Avg Risk'))
        fig.update_layout(title="Average Risk by Date", yaxis_title="Avg Risk (%)")
        st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        names = hist['name'].unique().tolist()
        selected_name = st.selectbox("View history for patient", ["All"] + names)
        if selected_name != "All":
            ph = hist[hist['name'] == selected_name]
            st.write(f"Records for {selected_name}:")
            st.dataframe(ph)
            ph['ts'] = pd.to_datetime(ph['timestamp'])
            ph_series = ph.sort_values('ts').set_index('ts')['risk']
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=ph_series.index, y=ph_series.values, mode='lines+markers', name=selected_name))
            fig2.update_layout(title=f"Risk trend — {selected_name}", yaxis_title="Risk (%)")
            st.plotly_chart(fig2, use_container_width=True)

# -------------------- REPORT & TIPS TAB --------------------
with tabs[3]:
    st.header("Practical Tips & Reports")
    st.write("Download the last saved report or generate a new one after prediction.")
    st.markdown("""
### Quick Lifestyle Tips
- Prefer vegetables, whole grains, legumes and lean protein.
- Reduce sugary beverages and processed snacks.
- Aim for 30–45 minutes of moderate exercise most days.
- Monitor BP and glucose; attend regular checkups.
    """)

# -------------------- STRESS & BREATHING TAB --------------------
with tabs[4]:
    st.header("Stress Meter & Guided Breathing")
    st.write("Slide to rate your current stress. Try a guided breathing exercise to calm down.")
    stress = st.slider("Current stress level (1 low — 10 high)", 1, 10, 4, key="stress_slider")
    st.write(f"Your stress level: **{stress}**")

    if stress <= 3:
        st.success("Low stress — keep it up!")
    elif stress <= 6:
        st.info("Moderate stress — try a short breathing break.")
    else:
        st.warning("High stress — consider a longer relaxation or professional support.")

    # Animated breathing ball (designer)
    st.markdown("<div class='breath-ball'></div>", unsafe_allow_html=True)
    st.markdown("### Guided 1-minute Box Breathing")
    if st.button("Start 60s Breathing"):
        placeholder = st.empty()
        cycles = 60 // 16 + 1
        for c in range(cycles):
            for t in range(4, 0, -1):
                placeholder.markdown(f"**Breathe in** — {t}s")
                time.sleep(1)
            for t in range(4, 0, -1):
                placeholder.markdown(f"**Hold** — {t}s")
                time.sleep(1)
            for t in range(4, 0, -1):
                placeholder.markdown(f"**Breathe out** — {t}s")
                time.sleep(1)
            for t in range(4, 0, -1):
                placeholder.markdown(f"**Hold** — {t}s")
                time.sleep(1)
        placeholder.markdown("✅ **Done — feel calmer?**")

# -------------------- COMPANION (DOCTOR) TAB --------------------
with tabs[5]:
    st.header("Doctor Companion — Symptom Triage (rule-based)")
    st.write("Type symptoms (comma-separated) or pick from quick checkboxes. This is a rule-based triage tool — for emergencies call local services.")
    chest_pain = st.checkbox("Chest pain or tightness")
    short_breath = st.checkbox("Shortness of breath")
    slurred = st.checkbox("Slurred speech or confusion")
    weakness = st.checkbox("Sudden weakness/numbness (face/arm/leg)")
    headache = st.checkbox("Severe sudden headache")
    other_txt = st.text_input("Other symptoms (comma-separated)")
    if st.button("Triage Symptoms"):
        urgent = False
        messages = []
        if chest_pain or short_breath:
            messages.append("⚠️ Chest pain or breathing trouble — possible cardiac emergency. Call emergency services immediately.")
            urgent = True
        if slurred or weakness or headache:
            messages.append("⚠️ Neurological signs (speech/weakness/headache) — possible stroke/TIA. Seek emergency care now (time-sensitive).")
            urgent = True
        if not urgent:
            if other_txt.strip():
                messages.append("Info: Review symptoms with your primary care physician within 24-72 hours.")
            else:
                messages.append("No immediate red-flag symptoms selected. If concerned, schedule a clinic visit or teleconsult.")
        for m in messages:
            if "⚠️" in m:
                st.error(m)
            else:
                st.info(m)

# -------------------- GENETIC RISK TAB --------------------
with tabs[6]:
    st.header("Simple Genetic-Risk Heuristic")
    st.write("This module uses user-provided family history and variant counts. It is a heuristic — true genetic risk requires lab/geneticist interpretation.")
    fam_hist = st.checkbox("First-degree family history of stroke/early CVD?")
    var_count = st.number_input("Number of known high-risk variants (user-provided)", 0, 100, 0)
    ethnicity_note = st.selectbox("Ethnicity (helps interpretation, optional)", ["Not specified","South Asian","European","African","East Asian","Other"])
    if st.button("Estimate Genetic Contribution"):
        base = 0.05
        bump = 0.08 if fam_hist else 0.0
        var_effect = 1 - np.exp(-var_count / 10)
        var_bump = 0.15 * var_effect
        genetic_risk_fraction = base + bump + var_bump
        genetic_risk_percent = round(genetic_risk_fraction * 100, 2)
        st.write(f"Estimated genetic contribution to baseline risk: **{genetic_risk_percent}%**")
        st.write("Interpretation notes:")
        st.write("- This is a heuristic. For precise genetic risk, consult a genetic counselor.")
        if fam_hist:
            st.write("- Family history increases probability of inherited risk factors; consider targeted screening.")
        if var_count > 0:
            st.write(f"- {var_count} variant(s) reported: consider sharing variant details with a clinician.")
        st.caption("Genetic risk interacts with environment and clinical factors; it's not destiny.")

# -------------------- ABOUT TAB --------------------
# -------------------- ABOUT TAB --------------------
with tabs[7]:

    # App Info Card
    st.markdown(
        """
        <div style="
            background: linear-gradient(145deg, #E8FDFB, #FFFFFF);
            padding: 28px;
            border-radius: 18px;
            border: 1px solid #A8E6E3;
            box-shadow: 0 8px 20px rgba(0,150,150,0.15);
            color:#024646;
        ">
        <h2 style="text-align:center; color:#008D8D; font-weight:800;">💡 About This Health Assistant</h2>

        <p style="font-size:16px; text-align:center; color:#056B6B;">
        A smart wellness tool built to guide, educate and help monitor stroke-risk and lifestyle health.
        </p>

        <hr style="border:1px solid #D4F7F5;">

        <h3 style="color:#009998;">✅ What this App Does</h3>
        <ul style="font-size:15px; line-height:1.65; margin-left:10px;">
            <li>Estimates <b>stroke risk</b> using ML</li>
            <li>Provides <b>personalized lifestyle advice</b></li>
            <li>Shows risk charts & <b>trend dashboards</b></li>
            <li>Saves reports for <b>progress tracking</b></li>
            <li>Offers <b>stress-relief tools</b></li>
            <li>Includes <b>emergency symptom triage</b></li>
            <li>Supports <b>genetic risk awareness</b></li>
            <li>Exports <b>PDF & text reports</b></li>
            <li>Explains <b>model reasoning</b></li>
        </ul>

        <hr style="border:1px solid #D4F7F5;">

        <h3 style="color:#009998;">🎯 Goal of the App</h3>
        <ul style="font-size:15px; line-height:1.65; margin-left:10px;">
            <li>Promote early stroke & health awareness</li>
            <li>Encourage healthier daily habits</li>
            <li>Identify potential warning signs early</li>
            <li>Make preventive health insights accessible</li>
        </ul>

        <hr style="border:1px solid #D4F7F5;">

        <h3 style="color:#CC5A5A;">⚠️ Important Notes</h3>
        <ul style="font-size:15px; line-height:1.65; margin-left:10px;">
            <li>This tool is for <b>awareness & wellness only</b></li>
            <li>Not a replacement for medical consultation</li>
            <li>Always consult healthcare professionals</li>
        </ul>

        <hr style="border:1px solid #D4F7F5;">

        <p style="text-align:center; font-size:14px; color:#0A7070; font-weight:600;">
        Empowering healthier lives — one informed decision at a time 🌿
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Creator Section ----
    st.write("")
    st.markdown("### 👤 About the Creator")

    # Creator 1
    creator_html = """
    <div style="
        padding: 18px;
        border-radius: 16px;
        background: linear-gradient(135deg, #eafff9, #c4fff0);
        border: 1px solid #00ccb4;
        box-shadow: 0 4px 14px rgba(0, 200, 170, 0.15);
        margin-bottom: 12px;
    ">
    <table style="width:100%;">
    <tr>
    <td style="width:120px; vertical-align:top; padding-right:15px;">
    <img src="https://media.licdn.com/dms/image/v2/D4D03AQFlS6DAPY4sKA/profile-displayphoto-shrink_400_400/B4DZV3ObW5HwAg-/0/1741462027428?e=1763596800&v=beta&t=zq1Rl_vFuVyebKxabjnokRevAjEJqCdmoj89MzSdJMQ"
         width="110" style="border-radius:12px; border:3px solid #00c9b7;">
    </td>

    <td style="vertical-align:top;">
    <p style="margin:0; font-size:22px; font-weight:800; color:#007064;">Anuya Janrao</p>
    <p style="margin:0; font-size:14px; color:#00463f;">
    Bioinformatician | AI & Healthcare | Genomics Research<br>
    Passionate about building intelligent tools for healthcare.
    </p>
    <br>
    <a href="https://www.linkedin.com/in/anuya-janrao-a6705b322" target="_blank" style="text-decoration:none;">
    <div style="
        background:#00d8c4; padding:8px 14px; border-radius:8px;
        color:white; font-weight:bold; display:inline-block; font-size:14px;
        box-shadow:0 3px 8px rgba(0,255,210,0.25);
    ">🔗 LinkedIn Profile</div></a>
    </td>
    </tr>
    </table>
    </div>
    """
    st.markdown(creator_html, unsafe_allow_html=True)

    # Creator 2
    creator_html = """
    <div style="
        padding: 18px;
        border-radius: 16px;
        background: linear-gradient(135deg, #eafff9, #c4fff0);
        border: 1px solid #00ccb4;
        box-shadow: 0 4px 14px rgba(0, 200, 170, 0.15);
        margin-bottom: 12px;
    ">
    <table style="width:100%;">
    <tr>
    <td style="width:120px; vertical-align:top; padding-right:15px;">
    <img src="https://media.licdn.com/dms/image/v2/D4D35AQFK3ecEGJQdyg/profile-framedphoto-shrink_400_400/B4DZmJpkSNIEAc-/0/1758950999457?e=1762509600&v=beta&t=ojFbvaH0XbtBeNDwxep3xwNcMWnjjDdEaZKHUXX_ce0"
         width="110" style="border-radius:12px; border:3px solid #00c9b7;">
    </td>

    <td style="vertical-align:top;">
    <p style="margin:0; font-size:22px; font-weight:800; color:#007064;">Arati Joshi</p>
    <p style="margin:0; font-size:14px; color:#00463f;">
    Bioinformatician | AI & Genomics Research <br>
    Turning biological data into meaningful healthcare insights.
    </p>
    <br>
    <a href="https://www.linkedin.com/in/arati-joshi-a2719819a" target="_blank" style="text-decoration:none;">
    <div style="
        background:#00d8c4; padding:8px 14px; border-radius:8px;
        color:white; font-weight:bold; display:inline-block; font-size:14px;
        box-shadow:0 3px 8px rgba(0,255,210,0.25);
    ">🔗 LinkedIn Profile</div></a>
    </td>
    </tr>
    </table>
    </div>
    """
    st.markdown(creator_html, unsafe_allow_html=True)

    # ---- Mentorship Section ----
    st.write("")
    st.markdown("### 👨‍🏫 Mentorship")

    mentor_html = """
    <div style="
        padding: 18px;
        border-radius: 16px;
        background: linear-gradient(135deg, #eafff9, #c4fff0);
        border: 1px solid #00ccb4;
        box-shadow: 0 4px 14px rgba(0, 200, 170, 0.15);
    ">
    <table style="width:100%;">
    <tr>
    <td style="width:120px; vertical-align:top; padding-right:15px;">
    <img src="https://media.licdn.com/dms/image/v2/D5603AQF9gsU7YBjWVg/profile-displayphoto-shrink_400_400/B56ZZI.WrdH0Ag-/0/1744981029051?e=1763596800&v=beta&t=bg8fuJhWkj7jan3iIqtA2-X63SIIZBXyWlbWcJw-nFY"
         width="110" style="border-radius:12px; border:3px solid #00c9b7;">
    </td>

    <td style="vertical-align:top;">
    <p style="margin:0; font-size:22px; font-weight:800; color:#007064;">Dr. Kushagra Kashyap</p>
    <p style="margin:0; font-size:14px; color:#00463f;">
    Assistant Professor, DES Pune University <br>
    Bioinformatics & Computational Biology Mentor
    </p>
    <br>
    <p style="margin:0; font-size:13px; color:#006158;">
    His mentorship provided direction in bioinformatics concepts, clinical alignment,
    and research validation for this digital health project.
    </p>
    <br>
    <a href="https://www.linkedin.com/in/dr-kushagra-kashyap-b230a3bb" target="_blank" style="text-decoration:none;">
    <div style="
        background:#00d8c4; padding:8px 14px; border-radius:8px;
        color:white; font-weight:bold; display:inline-block; font-size:14px;
        box-shadow:0 3px 8px rgba(0,255,210,0.25);
    ">🔗 LinkedIn Profile</div></a>
    </td>
    </tr>
    </table>
    </div>
    """
    st.markdown(mentor_html, unsafe_allow_html=True)
