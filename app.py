
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Load model & scaler ──────────────────────────────────────────
model  = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

features = [
    'age_years', 'gender', 'height', 'weight', 'bmi',
    'ap_hi', 'ap_lo', 'pulse_pressure',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]

# ── Konfigurasi halaman ──────────────────────────────────────────
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="🫀",
    layout="centered"
)

# ── Header ───────────────────────────────────────────────────────
st.title("🫀 Cardiovascular Disease Prediction")
st.markdown("Prediksi risiko penyakit kardiovaskular menggunakan Machine Learning.")
st.divider()

# ── Metrik Model (WAJIB sesuai ketentuan tugas) ──────────────────
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Model", "Random Forest")
col_m2.metric("Accuracy", "73.2%")
col_m3.metric("F1-Score", "0.73")
col_m4.metric("Dataset", "58K+ baris")
st.divider()

# ── Form Input ───────────────────────────────────────────────────
st.subheader("📋 Data Pasien")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Usia (tahun)", min_value=20, max_value=80, value=45)
    gender = st.selectbox("⚧ Jenis Kelamin", options=[1, 2], format_func=lambda x: "Perempuan" if x==1 else "Laki-laki")
    height = st.number_input("📏 Tinggi Badan (cm)", min_value=100, max_value=220, value=165)
    weight = st.number_input("⚖️ Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    ap_hi = st.number_input("💉 Tekanan Darah Sistolik (ap_hi)", min_value=60, max_value=240, value=120)
    ap_lo = st.number_input("💉 Tekanan Darah Diastolik (ap_lo)", min_value=40, max_value=160, value=80)

with col2:
    cholesterol = st.selectbox("🧪 Kolesterol", options=[1, 2, 3],
                                format_func=lambda x: {1:"Normal",2:"Di atas normal",3:"Jauh di atas normal"}[x])
    gluc = st.selectbox("🩸 Glukosa", options=[1, 2, 3],
                         format_func=lambda x: {1:"Normal",2:"Di atas normal",3:"Jauh di atas normal"}[x])
    smoke = st.selectbox("🚬 Merokok", options=[0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya")
    alco  = st.selectbox("🍺 Konsumsi Alkohol", options=[0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya")
    active = st.selectbox("🏃 Aktif Fisik", options=[0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya")

st.divider()

# ── Tombol Prediksi ──────────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary"):
    if ap_lo >= ap_hi:
        st.error("⚠️ Tekanan darah diastolik harus lebih kecil dari sistolik!")
    else:
        bmi = round(weight / ((height/100)**2), 2)
        pulse_pressure = ap_hi - ap_lo

        data_input = pd.DataFrame([[
            age, gender, height, weight, bmi,
            ap_hi, ap_lo, pulse_pressure,
            cholesterol, gluc, smoke, alco, active
        ]], columns=features)

        data_scaled = pd.DataFrame(scaler.transform(data_input), columns=features)
        prediction  = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0]

        st.subheader("📊 Hasil Prediksi")
        if prediction == 0:
            st.success("✅ TIDAK TERDETEKSI penyakit kardiovaskular")
        else:
            st.error("❌ TERDETEKSI risiko penyakit kardiovaskular")

        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Prob. Sehat", f"{probability[0]*100:.1f}%")
        col_p2.metric("Prob. Penyakit", f"{probability[1]*100:.1f}%")
        col_p3.metric("BMI", f"{bmi:.1f}")

        st.divider()
        if prediction == 1:
            if cholesterol == 3 or gluc == 3:
                st.warning("💡 Kolesterol/glukosa sangat tinggi. Konsultasikan ke dokter.")
            elif ap_hi >= 140:
                st.warning("💡 Tekanan darah tinggi (hipertensi). Perlu penanganan medis.")
            elif bmi >= 30:
                st.warning("💡 BMI termasuk kategori obesitas. Disarankan menurunkan berat badan.")
            else:
                st.warning("💡 Profil menunjukkan risiko kardiovaskular. Konsultasikan ke dokter.")
        else:
            st.info("💡 Profil menunjukkan kondisi sehat. Tetap jaga pola hidup aktif!")

st.divider()
st.caption("🎓 Final Project AI & Big Data 2026 | Cardiovascular Disease Prediction")
