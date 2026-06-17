import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioCheck",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A1628;
    color: #E8EDF5;
}

.stApp {
    background: linear-gradient(135deg, #0A1628 0%, #0D1F3C 50%, #0A1628 100%);
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 720px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 0.5rem;
}
.hero-icon {
    font-size: 3.5rem;
    line-height: 1;
    margin-bottom: 0.75rem;
    display: block;
    animation: pulse-icon 2.5s ease-in-out infinite;
}
@keyframes pulse-icon {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.08); }
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00D4AA, #4FC3F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #8899B4;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}

/* ── Section card ── */
.section-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00D4AA;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(0,212,170,0.2);
}

/* ── Derived values box ── */
.derived-box {
    background: rgba(0,212,170,0.06);
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
}
.derived-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    font-size: 0.88rem;
}
.derived-label { color: #8899B4; }
.derived-value { 
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: #00D4AA;
}
.derived-badge {
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 6px;
    font-weight: 500;
}
.badge-normal { background: rgba(46,204,113,0.15); color: #2ECC71; }
.badge-over   { background: rgba(255,165,0,0.15);  color: #FFA500; }
.badge-obese  { background: rgba(255,71,87,0.15);  color: #FF4757; }
.badge-under  { background: rgba(150,150,255,0.15); color: #9696FF; }

/* ── Inputs styling ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E8EDF5 !important;
    font-family: 'Inter', sans-serif !important;
}
.stNumberInput > div > div > input:focus,
.stSelectbox > div > div:focus {
    border-color: rgba(0,212,170,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.1) !important;
}
label, .stSelectbox label, .stNumberInput label {
    color: #B0BFCC !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00D4AA, #00A87F) !important;
    color: #0A1628 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(0,212,170,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,212,170,0.45) !important;
}

/* ── Result cards ── */
.result-risk {
    background: linear-gradient(135deg, rgba(255,71,87,0.12), rgba(255,71,87,0.05));
    border: 1px solid rgba(255,71,87,0.4);
    border-radius: 16px;
    padding: 1.75rem;
    text-align: center;
    animation: fadeInUp 0.5s ease;
}
.result-safe {
    background: linear-gradient(135deg, rgba(46,204,113,0.12), rgba(46,204,113,0.05));
    border: 1px solid rgba(46,204,113,0.4);
    border-radius: 16px;
    padding: 1.75rem;
    text-align: center;
    animation: fadeInUp 0.5s ease;
}
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0.5rem 0 0.25rem;
}
.result-prob {
    font-size: 2.8rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    line-height: 1.1;
    margin: 0.4rem 0;
}
.result-sub { color: #8899B4; font-size: 0.85rem; }

/* ── Gauge bar ── */
.gauge-wrap { margin: 1.25rem 0 0.5rem; }
.gauge-track {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.gauge-fill-risk {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #FF9A3C, #FF4757);
    transition: width 1s cubic-bezier(.4,0,.2,1);
}
.gauge-fill-safe {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #00D4AA, #2ECC71);
    transition: width 1s cubic-bezier(.4,0,.2,1);
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 4px;
    font-size: 0.72rem;
    color: #6677A0;
}

/* ── Override notice ── */
.override-notice {
    background: rgba(255,165,0,0.08);
    border: 1px solid rgba(255,165,0,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.85rem;
}
.override-title {
    color: #FFA500;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.override-item {
    color: #C8B89A;
    padding: 2px 0;
    display: flex;
    align-items: flex-start;
    gap: 6px;
}

/* ── Prob detail table ── */
.prob-table {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.prob-cell {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.prob-cell-label { font-size: 0.75rem; color: #8899B4; margin-bottom: 4px; }
.prob-cell-val { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; }
.prob-safe-val { color: #2ECC71; }
.prob-risk-val { color: #FF4757; }

/* ── Recommendation ── */
.reco-box {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #00D4AA;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin-top: 1rem;
    font-size: 0.88rem;
    color: #A8B8CC;
    line-height: 1.6;
}
.reco-box b { color: #E8EDF5; }

/* ── Divider ── */
.custom-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    model  = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_models()

FEATURE_COLS = [
    'age_years', 'gender', 'height', 'weight',
    'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active',
    'BMI', 'pulse_pressure', 'MAP', 'BMI_category'
]

# ── Clinical safety override ─────────────────────────────────────────
def check_clinical_risk(ap_hi, ap_lo, chol_val, gluc_val, age, bmi_val):
    reasons = []
    if ap_hi >= 180:
        reasons.append(f"Tekanan sistolik {ap_hi} mmHg (Krisis Hipertensi)")
    elif ap_hi >= 140:
        reasons.append(f"Tekanan sistolik {ap_hi} mmHg (Hipertensi Stage 2)")
    if ap_lo >= 120:
        reasons.append(f"Tekanan diastolik {ap_lo} mmHg (Krisis Hipertensi)")
    elif ap_lo >= 90:
        reasons.append(f"Tekanan diastolik {ap_lo} mmHg (Hipertensi)")
    if chol_val == 3:
        reasons.append("Kolesterol jauh di atas normal")
    if gluc_val == 3:
        reasons.append("Glukosa jauh di atas normal (indikasi diabetes)")
    if age >= 55 and (ap_hi >= 130 or chol_val >= 2):
        reasons.append(f"Usia {age} tahun dengan faktor risiko tambahan")
    if bmi_val >= 35:
        reasons.append(f"Obesitas berat (BMI {bmi_val:.1f})")
    if ap_hi >= 140 and chol_val >= 2:
        reasons.append("Hipertensi + kolesterol tinggi (kombinasi berbahaya)")
    if ap_hi >= 140 and gluc_val >= 2:
        reasons.append("Hipertensi + gangguan gula darah")
    is_high_risk = len(reasons) >= 2 or ap_hi >= 180 or ap_lo >= 120
    return is_high_risk, reasons

# ── HERO ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">🫀</span>
    <h1 class="hero-title">CardioCheck</h1>
    <p class="hero-sub">Analisis risiko kardiovaskular berbasis machine learning</p>
</div>
""", unsafe_allow_html=True)

# ── FORM ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📋 &nbsp;Data Dasar</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age    = st.number_input("Umur (tahun)", min_value=1, max_value=120, value=45)
    height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=165)
    weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=200.0, value=70.0, step=0.5)
with col2:
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    ap_hi  = st.number_input("Tekanan Darah Sistolik", min_value=60, max_value=250, value=120)
    ap_lo  = st.number_input("Tekanan Darah Diastolik", min_value=40, max_value=200, value=80)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">🧪 &nbsp;Hasil Lab & Gaya Hidup</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    cholesterol = st.selectbox("Kolesterol", ["Normal", "Di atas normal", "Jauh di atas normal"])
    gluc        = st.selectbox("Glukosa Darah", ["Normal", "Di atas normal", "Jauh di atas normal"])
with col4:
    smoke  = st.selectbox("Merokok?", ["Tidak", "Ya"])
    alco   = st.selectbox("Konsumsi Alkohol?", ["Tidak", "Ya"])
    active = st.selectbox("Aktif Berolahraga?", ["Tidak", "Ya"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Derived values display ────────────────────────────────────────────
bmi_val = weight / ((height / 100) ** 2)
pp_val  = ap_hi - ap_lo
map_val = (ap_hi + 2 * ap_lo) / 3
if bmi_val < 18.5:   bmi_cat, bmi_label, bmi_badge = 0, "Underweight", "badge-under"
elif bmi_val < 25:   bmi_cat, bmi_label, bmi_badge = 1, "Normal",      "badge-normal"
elif bmi_val < 30:   bmi_cat, bmi_label, bmi_badge = 2, "Overweight",  "badge-over"
else:                bmi_cat, bmi_label, bmi_badge = 3, "Obese",       "badge-obese"

bp_status = "Normal"
bp_badge  = "badge-normal"
if ap_hi >= 180 or ap_lo >= 120:  bp_status, bp_badge = "Krisis!", "badge-obese"
elif ap_hi >= 140 or ap_lo >= 90: bp_status, bp_badge = "Tinggi",  "badge-over"
elif ap_hi >= 130 or ap_lo >= 80: bp_status, bp_badge = "Elevated","badge-over"

st.markdown(f"""
<div class="derived-box">
    <div style="font-size:0.75rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#8899B4;margin-bottom:0.6rem;">
        Nilai Turunan
    </div>
    <div class="derived-row">
        <span class="derived-label">Body Mass Index (BMI)</span>
        <span>
            <span class="derived-value">{bmi_val:.1f}</span>
            <span class="derived-badge {bmi_badge}">{bmi_label}</span>
        </span>
    </div>
    <div class="derived-row">
        <span class="derived-label">Pulse Pressure</span>
        <span class="derived-value">{pp_val} mmHg</span>
    </div>
    <div class="derived-row">
        <span class="derived-label">Mean Arterial Pressure (MAP)</span>
        <span>
            <span class="derived-value">{map_val:.1f} mmHg</span>
            <span class="derived-badge {bp_badge}">{bp_status}</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────
predict_clicked = st.button("🔍  Analisis Sekarang", use_container_width=True)

# ── Result ────────────────────────────────────────────────────────────
if predict_clicked:
    chol_map   = {"Normal": 1, "Di atas normal": 2, "Jauh di atas normal": 3}
    gender_val = 1 if gender == "Laki-laki" else 2
    chol_val   = chol_map[cholesterol]
    gluc_val   = chol_map[gluc]
    smoke_val  = 1 if smoke == "Ya" else 0
    alco_val   = 1 if alco  == "Ya" else 0
    active_val = 1 if active == "Ya" else 0

    input_data = pd.DataFrame([[
        float(age), gender_val, float(height), float(weight),
        float(ap_hi), float(ap_lo), chol_val, gluc_val,
        smoke_val, alco_val, active_val,
        bmi_val, float(pp_val), map_val, bmi_cat
    ]], columns=FEATURE_COLS)

    input_scaled    = scaler.transform(input_data)
    input_scaled[0][1] = float(gender_val)   # fix NaN gender scaler

    prediction = model.predict(input_scaled)[0]
    proba      = model.predict_proba(input_scaled)[0]

    clinical_high_risk, risk_reasons = check_clinical_risk(
        ap_hi, ap_lo, chol_val, gluc_val, age, bmi_val
    )

    model_overridden = False
    if prediction == 0 and clinical_high_risk:
        prediction       = 1
        model_overridden = True

    risk_pct = proba[1] * 100
    safe_pct = proba[0] * 100

    if prediction == 1:
        gauge_pct   = max(risk_pct, 75) if model_overridden else risk_pct
        gauge_class = "gauge-fill-risk"
        emoji        = "⚠️"
        result_class = "result-risk"
        label        = "BERISIKO TINGGI"
        label_color  = "#FF4757"
        reco = ("<b>Segera konsultasikan dengan dokter.</b> "
                "Pantau tekanan darah secara rutin, terapkan pola makan rendah garam & lemak, "
                "dan tingkatkan aktivitas fisik secara bertahap.")
    else:
        gauge_pct   = safe_pct
        gauge_class = "gauge-fill-safe"
        emoji        = "✅"
        result_class = "result-safe"
        label        = "RISIKO RENDAH"
        label_color  = "#2ECC71"
        reco = ("<b>Tetap pertahankan gaya hidup sehat.</b> "
                "Lakukan pemeriksaan rutin minimal 1x setahun, jaga berat badan ideal, "
                "dan hindari rokok serta alkohol berlebihan.")

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="{result_class}">
        <div style="font-size:2rem">{emoji}</div>
        <div class="result-title" style="color:{label_color}">{label}</div>
        <div class="result-sub">Prediksi berdasarkan data yang dimasukkan</div>
        <div class="gauge-wrap">
            <div class="gauge-track">
                <div class="{gauge_class}" style="width:{min(gauge_pct,100):.1f}%"></div>
            </div>
            <div class="gauge-labels"><span>Aman</span><span>Berisiko</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Override notice
    if model_overridden:
        items_html = "".join([
            f'<div class="override-item"><span>⚡</span><span>{r}</span></div>'
            for r in risk_reasons
        ])
        st.markdown(f"""
        <div class="override-notice">
            <div class="override-title">⚠️ Deteksi Klinis Aktif</div>
            <div style="color:#C8B89A;font-size:0.82rem;margin-bottom:0.5rem;">
                Model menunjukkan risiko rendah, namun indikator medis berikut terdeteksi:
            </div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

    # Probability detail
    st.markdown(f"""
    <div class="prob-table">
        <div class="prob-cell">
            <div class="prob-cell-label">Probabilitas Aman</div>
            <div class="prob-cell-val prob-safe-val">{safe_pct:.1f}%</div>
        </div>
        <div class="prob-cell">
            <div class="prob-cell-label">Probabilitas Risiko</div>
            <div class="prob-cell-val prob-risk-val">{risk_pct:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation
    st.markdown(f'<div class="reco-box">💡 <b>Rekomendasi:</b> {reco}</div>', unsafe_allow_html=True)
