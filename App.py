import streamlit as st
import numpy as np
import joblib  # untuk memuat model yang disimpan

# Memuat model prediksi
rf_model = joblib.load('best_random_forest_model.pkl')  # Model Random Forest
svm_model = joblib.load('svm_model_hyperparameter_tuned.pkl')  # Model SVM
knn_model = joblib.load('knn_model_hyperparameter_tuned.pkl')  # Model KNN

# Fungsi prediksi
def predict(model, AccountWeeks, ContractRenewal, DataUsage, CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins):
    result = model.predict([[AccountWeeks, ContractRenewal, DataUsage, CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins]])
    return result[0]

# Header
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #2980B9;
        margin-bottom: 30px;
        font-family: 'Arial', sans-serif;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #16A085;
        margin-bottom: 20px;
        font-family: 'Arial', sans-serif;
    }
    .prediction-header {
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        color: #D35400;
        margin-bottom: 15px;
        font-family: 'Arial', sans-serif;
    }
    .result-box {
        background-color: #ECF0F1;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #888;
        font-size: 1rem;
        font-family: 'Arial', sans-serif;
    }
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .sidebar-header {
            font-size: 1.1rem;
        }
        .prediction-header {
            font-size: 1.4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="main-header">✨ Prediksi Pelanggan Berlangganan ✨</div>', unsafe_allow_html=True)

# Layout input di sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">📝 Input Data Pelanggan</div>', unsafe_allow_html=True)
    
    # Penjelasan untuk setiap input
    st.markdown("**1. Account Weeks**: Jumlah minggu pelanggan telah aktif menggunakan layanan.")
    AccountWeeks = st.number_input("Account Weeks", min_value=1, max_value=1000, value=128)

    st.markdown("**2. Contract Renewal**: Status pembaruan kontrak pelanggan. (0 = Tidak, 1 = Ya)")
    ContractRenewal = st.selectbox("Contract Renewal", options=[0, 1])

    st.markdown("**3. Data Usage (GB)**: Total penggunaan data pelanggan dalam GB.")
    DataUsage = st.number_input("Data Usage (GB)", min_value=0.0, value=2.70, format="%.2f")

    st.markdown("**4. Customer Service Calls**: Jumlah panggilan pelanggan ke layanan pelanggan.")
    CustServCalls = st.number_input("Customer Service Calls", min_value=0, max_value=50, value=1)

    st.markdown("**5. Day Minutes**: Total durasi panggilan telepon pelanggan dalam menit pada siang hari.")
    DayMins = st.number_input("Day Minutes", min_value=0.0, value=265.1, format="%.1f")

    st.markdown("**6. Day Calls**: Jumlah panggilan telepon pelanggan pada siang hari.")
    DayCalls = st.number_input("Day Calls", min_value=0, max_value=500, value=110)

    st.markdown("**7. Monthly Charge ($)**: Biaya bulanan yang dikenakan kepada pelanggan.")
    MonthlyCharge = st.number_input("Monthly Charge ($)", min_value=0.0, value=89.0, format="%.2f")

    st.markdown("**8. Overage Fee ($)**: Biaya tambahan yang dikenakan jika pelanggan melewati batas penggunaan layanan.")
    OverageFee = st.number_input("Overage Fee ($)", min_value=0.0, value=9.87, format="%.2f")

    st.markdown("**9. Roaming Minutes**: Total durasi penggunaan layanan dalam menit saat pelanggan berada di luar area layanan utama.")
    RoamMins = st.number_input("Roaming Minutes", min_value=0.0, value=10.0, format="%.1f")

# Bagian Informasi
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        **Jenis Algoritma yang Digunakan:**
        - Random Forest
        - Support Vector Machine
        - K-Nearest Neighbour
        """
    )

with col2:
    st.markdown(
        """
        **Tentang Aplikasi**  
        Aplikasi ini memprediksi apakah pelanggan akan tetap berlangganan atau berhenti berlangganan berdasarkan data pengguna.
        """
    )

st.markdown("---")

# Tombol Prediksi
if st.button("🔮 Prediksi Sekarang!"):
    # Prediksi dengan semua model
    rf_prediction = predict(rf_model, AccountWeeks, ContractRenewal, DataUsage, CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins)
    svm_prediction = predict(svm_model, AccountWeeks, ContractRenewal, DataUsage, CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins)
    knn_prediction = predict(knn_model, AccountWeeks, ContractRenewal, DataUsage, CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins)

    # Tampilkan hasil prediksi
    st.markdown('<div class="prediction-header">Hasil Prediksi dari Model</div>', unsafe_allow_html=True)

    # Random Forest
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    if rf_prediction == 0:
        st.success("🌳 **Random Forest** memprediksi: ✅ Pelanggan akan **bertahan berlangganan**.")
    else:
        st.error("🌳 **Random Forest** memprediksi: ⚠️ Pelanggan akan **berhenti berlangganan**.")
    st.markdown('</div>', unsafe_allow_html=True)

    # SVM
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    if svm_prediction == 0:
        st.success("🧩 **Support Vector Machine** memprediksi: ✅ Pelanggan akan **bertahan berlangganan**.")
    else:
        st.error("🧩 **Support Vector Machine** memprediksi: ⚠️ Pelanggan akan **berhenti berlangganan**.")
    st.markdown('</div>', unsafe_allow_html=True)

    # KNN
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    if knn_prediction == 0:
        st.success("🤝 **K-Nearest Neighbour** memprediksi: ✅ Pelanggan akan **bertahan berlangganan**.")
    else:
        st.error("🤝 **K-Nearest Neighbour** memprediksi: ⚠️ Pelanggan akan **berhenti berlangganan**.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("**Keterangan:**")
    st.markdown("- **0** : Pelanggan akan bertahan berlangganan")
    st.markdown("- **1** : Pelanggan akan berhenti berlangganan")

# Footer
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #888;
        font-size: 1rem;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="footer">Dibuat dengan ❤️ oleh Yohanes J Palis</div>', unsafe_allow_html=True)
