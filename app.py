import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Load model & scaler ──────────────────────────────────────────────
model  = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

FEATURE_COLS = [
    'age_years', 'gender', 'height', 'weight',
    'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active',
    'BMI', 'pulse_pressure', 'MAP', 'BMI_category'
]

MODEL_ACCURACY = "73.00%"
MODEL_F1       = "0.7300"

# ── Rule-based safety check (medical override) ───────────────────────
def check_clinical_risk(ap_hi, ap_lo, cholesterol_val, gluc_val, age, bmi_val):
    """
    Cek kondisi klinis yang secara medis jelas berisiko tinggi.
    Mengembalikan (is_high_risk: bool, reasons: list[str])
    """
    reasons = []

    # Hipertensi Stage 2+ (sistolik ≥140 ATAU diastolik ≥90)
    if ap_hi >= 180:
        reasons.append(f"Tekanan darah sistolik sangat tinggi ({ap_hi} mmHg — Krisis Hipertensi)")
    elif ap_hi >= 140:
        reasons.append(f"Tekanan darah sistolik tinggi ({ap_hi} mmHg — Hipertensi Stage 2)")

    if ap_lo >= 120:
        reasons.append(f"Tekanan darah diastolik sangat tinggi ({ap_lo} mmHg — Krisis Hipertensi)")
    elif ap_lo >= 90:
        reasons.append(f"Tekanan darah diastolik tinggi ({ap_lo} mmHg — Hipertensi)")

    # Kolesterol jauh di atas normal
    if cholesterol_val == 3:
        reasons.append("Kolesterol jauh di atas normal")

    # Glukosa jauh di atas normal (indikasi diabetes)
    if gluc_val == 3:
        reasons.append("Glukosa jauh di atas normal (kemungkinan diabetes)")

    # Usia + kombinasi faktor risiko
    if age >= 55 and (ap_hi >= 130 or cholesterol_val >= 2):
        reasons.append(f"Usia {age} tahun dengan faktor risiko tambahan")

    # Obesitas berat
    if bmi_val >= 35:
        reasons.append(f"Obesitas berat (BMI {bmi_val:.1f})")

    # Kombinasi berbahaya: hipertensi + kolesterol tinggi
    if ap_hi >= 140 and cholesterol_val >= 2:
        reasons.append("Kombinasi hipertensi + kolesterol tinggi (risiko sangat tinggi)")

    # Kombinasi: hipertensi + diabetes
    if ap_hi >= 140 and gluc_val >= 2:
        reasons.append("Kombinasi hipertensi + gangguan gula darah")

    is_high_risk = len(reasons) >= 2 or ap_hi >= 180 or ap_lo >= 120
    return is_high_risk, reasons


# ── Halaman config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ── Header ───────────────────────────────────────────────────────────
st.title("❤️ Cardiovascular Disease Prediction")
st.markdown(
    "Masukkan data kesehatan pasien untuk memprediksi risiko "
    "**penyakit kardiovaskular**."
)
st.divider()

# ── Info performa model ───────────────────────────────────────────────
col_a, col_b = st.columns(2)
col_a.metric("Model Accuracy", MODEL_ACCURACY)
col_b.metric("F1-Score", MODEL_F1)
st.caption("Model: Decision Tree Classifier | Dataset: Cardiovascular Disease (Kaggle, 70.000 data)")
st.divider()

# ── Input Form ───────────────────────────────────────────────────────
st.subheader("📋 Data Pasien")

col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Umur (tahun)", min_value=1, max_value=120, value=45)
    gender   = st.selectbox("Jenis Kelamin", options=["Perempuan", "Laki-laki"])
    height   = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=165)
    weight   = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=200.0, value=70.0, step=0.5)
    ap_hi    = st.number_input("Tekanan Darah Sistolik (ap_hi)", min_value=60, max_value=250, value=120)
    ap_lo    = st.number_input("Tekanan Darah Diastolik (ap_lo)", min_value=40, max_value=200, value=80)
    cholesterol = st.selectbox("Kolesterol", options=["Normal (1)", "Di atas normal (2)", "Jauh di atas normal (3)"])
    gluc     = st.selectbox("Glukosa", options=["Normal (1)", "Di atas normal (2)", "Jauh di atas normal (3)"])

with col2:
    smoke    = st.selectbox("Merokok?", options=["Tidak", "Ya"])
    alco     = st.selectbox("Konsumsi Alkohol?", options=["Tidak", "Ya"])
    active   = st.selectbox("Aktif Berolahraga?", options=["Tidak", "Ya"])

    # Hitung fitur turunan
    bmi_val = weight / ((height / 100) ** 2)
    pp_val  = ap_hi - ap_lo
    map_val = (ap_hi + 2 * ap_lo) / 3
    if bmi_val < 18.5:   bmi_cat = 0
    elif bmi_val < 25:   bmi_cat = 1
    elif bmi_val < 30:   bmi_cat = 2
    else:                bmi_cat = 3
    bmi_labels = {0: "Underweight", 1: "Normal", 2: "Overweight", 3: "Obese"}

    st.markdown("**📊 Nilai Turunan (otomatis dihitung):**")
    st.info(
        f"BMI: **{bmi_val:.1f}** ({bmi_labels[bmi_cat]})  \n"
        f"Pulse Pressure: **{pp_val}**  \n"
        f"MAP: **{map_val:.1f}**"
    )

st.divider()

# ── Prediksi ─────────────────────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary"):

    # Konversi input
    gender_val = 1 if gender == "Laki-laki" else 2
    chol_map   = {"Normal (1)": 1, "Di atas normal (2)": 2, "Jauh di atas normal (3)": 3}
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

    # Scale input — fix NaN pada kolom gender
    input_scaled = scaler.transform(input_data)
    input_scaled[0][1] = float(gender_val)   # gender tidak ter-scale dengan benar saat training

    prediction = model.predict(input_scaled)[0]
    proba      = model.predict_proba(input_scaled)[0]

    # ── Clinical safety check ────────────────────────────────────────
    clinical_high_risk, risk_reasons = check_clinical_risk(
        ap_hi, ap_lo, chol_val, gluc_val, age, bmi_val
    )

    # Jika model bilang AMAN tapi secara klinis jelas berisiko → override
    model_overridden = False
    if prediction == 0 and clinical_high_risk:
        prediction = 1
        model_overridden = True

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.error(
            f"⚠️ **BERISIKO** terkena penyakit kardiovaskular\n\n"
            f"Probabilitas risiko model: **{proba[1]*100:.1f}%**"
        )

        if model_overridden:
            st.warning(
                "⚠️ **Catatan:** Prediksi model awal menunjukkan 'tidak berisiko', "
                "namun **indikator klinis** menunjukkan adanya faktor risiko signifikan "
                "yang terdeteksi secara medis:"
            )
            for r in risk_reasons:
                st.markdown(f"  - {r}")

        st.markdown(
            "**Rekomendasi:** Segera konsultasikan dengan dokter, "
            "jaga pola makan, dan rutin berolahraga."
        )
    else:
        st.success(
            f"✅ **TIDAK BERISIKO** terkena penyakit kardiovaskular\n\n"
            f"Probabilitas aman: **{proba[0]*100:.1f}%**"
        )

        if risk_reasons:
            st.warning(
                "⚠️ Meskipun prediksi aman, perhatikan faktor berikut:"
            )
            for r in risk_reasons:
                st.markdown(f"  - {r}")

        st.markdown(
            "**Rekomendasi:** Tetap jaga gaya hidup sehat dan "
            "lakukan pemeriksaan rutin."
        )

    # Detail probabilitas
    st.markdown("**Detail probabilitas dari model:**")
    prob_df = pd.DataFrame({
        'Kelas': ['Tidak Berisiko (0)', 'Berisiko (1)'],
        'Probabilitas': [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"]
    })
    st.table(prob_df)

# ── Footer ───────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Disclaimer: Prediksi ini hanya untuk keperluan akademik "
    "dan tidak menggantikan diagnosis medis profesional."
)
