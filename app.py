import streamlit as st
import joblib
import pandas as pd

# ── Load model (Decision Tree, no scaler needed) ─────────────────
model = joblib.load('best_model.joblib')

FEATURES = [
    'age_years', 'gender', 'height', 'weight',
    'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active',
    'BMI', 'pulse_pressure', 'MAP', 'BMI_category'
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

# ── Metrik Model ─────────────────────────────────────────────────
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Model", "Decision Tree")
col_m2.metric("Accuracy", "72.4%")
col_m3.metric("F1-Score", "0.72")
col_m4.metric("Dataset", "70K baris")
st.divider()

# ── Form Input ───────────────────────────────────────────────────
st.subheader("📋 Data Pasien")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Usia (tahun)", min_value=20, max_value=80, value=45)
    gender = st.selectbox(
        "⚧ Jenis Kelamin",
        options=[1, 2],
        format_func=lambda x: "Perempuan" if x == 1 else "Laki-laki"
    )
    height = st.number_input("📏 Tinggi Badan (cm)", min_value=100, max_value=220, value=165)
    weight = st.number_input("⚖️ Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    ap_hi = st.number_input("💉 Tekanan Darah Sistolik", min_value=60, max_value=240, value=120)
    ap_lo = st.number_input("💉 Tekanan Darah Diastolik", min_value=40, max_value=160, value=80)

with col2:
    cholesterol = st.selectbox(
        "🧪 Kolesterol",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Di atas normal", 3: "Jauh di atas normal"}[x]
    )
    gluc = st.selectbox(
        "🩸 Glukosa",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Di atas normal", 3: "Jauh di atas normal"}[x]
    )
    smoke  = st.selectbox("🚬 Merokok", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    alco   = st.selectbox("🍺 Konsumsi Alkohol", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    active = st.selectbox("🏃 Aktif Fisik", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

st.divider()

# ── Tombol Prediksi ──────────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary"):
    if ap_lo >= ap_hi:
        st.error("⚠️ Tekanan darah diastolik harus lebih kecil dari sistolik!")
    else:
        # Feature engineering (sama persis seperti saat training)
        age_years      = float(age)
        BMI            = round(weight / ((height / 100) ** 2), 2)
        pulse_pressure = ap_hi - ap_lo
        MAP            = round((ap_hi + 2 * ap_lo) / 3, 2)

        if BMI < 18.5:
            BMI_category = 0   # Underweight
        elif BMI < 25.0:
            BMI_category = 1   # Normal
        elif BMI < 30.0:
            BMI_category = 2   # Overweight
        else:
            BMI_category = 3   # Obese

        data_input = pd.DataFrame([[
            age_years, gender, height, weight,
            ap_hi, ap_lo, cholesterol, gluc,
            smoke, alco, active,
            BMI, pulse_pressure, MAP, BMI_category
        ]], columns=FEATURES)

        prediction  = model.predict(data_input)[0]
        probability = model.predict_proba(data_input)[0]

        st.subheader("📊 Hasil Prediksi")
        if prediction == 0:
            st.success("✅ TIDAK TERDETEKSI penyakit kardiovaskular")
        else:
            st.error("❌ TERDETEKSI risiko penyakit kardiovaskular")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prob. Sehat",    f"{probability[0]*100:.1f}%")
        c2.metric("Prob. Penyakit", f"{probability[1]*100:.1f}%")
        c3.metric("BMI",            f"{BMI:.1f}")
        c4.metric("MAP",            f"{MAP:.1f} mmHg")

        # Kategori BMI
        bmi_label = {0: "Underweight", 1: "Normal", 2: "Overweight", 3: "Obese"}
        st.caption(f"Kategori BMI: **{bmi_label[BMI_category]}** | Pulse Pressure: **{pulse_pressure} mmHg**")

        st.divider()
        if prediction == 1:
            if ap_hi >= 140:
                st.warning("💡 Tekanan darah tinggi (hipertensi). Konsultasikan ke dokter.")
            elif cholesterol == 3 or gluc == 3:
                st.warning("💡 Kolesterol/glukosa sangat tinggi. Perlu penanganan medis.")
            elif BMI >= 30:
                st.warning("💡 BMI masuk kategori obesitas. Disarankan menurunkan berat badan.")
            else:
                st.warning("💡 Profil menunjukkan risiko kardiovaskular. Konsultasikan ke dokter.")
        else:
            st.info("💡 Profil menunjukkan kondisi sehat. Tetap jaga pola hidup aktif!")

# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.caption("🎓 Final Project AI & Big Data 2026 | Cardiovascular Disease Prediction")
