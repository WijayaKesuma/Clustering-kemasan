
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title="Segmentasi Produk Kemasan", page_icon="📦", layout="centered")

st.title("Sistem Segmentasi Produk Kemasan")
st.caption("Deployment model K-Means Clustering berbasis RFM (Recency, Frequency, Monetary)")

# ---------------------------
# Load model, scaler, dan label
# ---------------------------
@st.cache_resource
def load_artifacts():
    missing = [f for f in ["kmeans_model.pkl", "scaler.pkl", "cluster_labels.json"] if not os.path.exists(f)]
    if missing:
        return None, None, None, missing
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("cluster_labels.json") as f:
        labels_data = json.load(f)
    return model, scaler, labels_data, []

model, scaler, labels_data, missing_files = load_artifacts()

if missing_files:
    st.error(
        "File model belum ditemukan: " + ", ".join(missing_files) +
        ". Silakan upload hasil dari Colab (Bagian A) ke folder yang sama dengan app.py."
    )
    st.stop()

cluster_names = labels_data["names"]
cluster_recs = labels_data["recommendations"]

tab1, tab2 = st.tabs(["🔮 Prediksi Segmen Produk Baru", "ℹTentang Sistem"])

# ---------------------------
# TAB 1: Prediksi
# ---------------------------
with tab1:
    st.subheader("Masukkan Data Histori Transaksi Produk")
    st.write("Isi nilai berikut berdasarkan histori transaksi produk yang ingin dianalisis:")

    col1, col2 = st.columns(2)
    with col1:
        recency = st.number_input("Recency (hari sejak transaksi terakhir)", min_value=0, value=30)
        frequency = st.number_input("Frequency (jumlah total transaksi)", min_value=1, value=5)
        monetary = st.number_input("Monetary (total omzet produk, Rp)", min_value=0, value=5_000_000, step=100_000)
    with col2:
        avg_order_qty = st.number_input("Avg Order Qty (rata-rata unit per transaksi)", min_value=0.0, value=1000.0)
        avg_price = st.number_input("Avg Price (rata-rata harga satuan, Rp)", min_value=0.0, value=1500.0)
        total_qty = st.number_input("Total Qty (total unit terjual)", min_value=0, value=5000)

    if st.button("🔍 Prediksi Segmen", type="primary"):
        # Transformasi sama persis seperti saat training (BAB III)
        raw = pd.DataFrame(
            [[recency, frequency, monetary, avg_order_qty, avg_price, total_qty]],
            columns=["Recency", "Frequency", "Monetary", "Avg_Order_Qty", "Avg_Price", "Total_Qty"],
        )
        log_transformed = np.log1p(raw)
        scaled = scaler.transform(log_transformed)

        cluster_pred = model.predict(scaled)[0]
        segment_name = cluster_names[str(cluster_pred)]
        recommendation = cluster_recs[str(cluster_pred)]

        st.success(f"**Cluster {cluster_pred}: {segment_name}**")
        st.info(f"**Rekomendasi strategi:** {recommendation}")

        with st.expander("Lihat detail hasil transformasi data"):
            st.write("Data input asli:")
            st.dataframe(raw)
            st.write("Data setelah standardisasi (siap dipakai model):")
            st.dataframe(pd.DataFrame(scaled, columns=raw.columns))

# ---------------------------
# TAB 2: Penjelasan sistem
# ---------------------------
with tab2:
    st.subheader("Alur Kerja Sistem")
    st.markdown("""
    1. **Input** — pengguna memasukkan data RFM produk (Recency, Frequency, Monetary, dll).
    2. **Transformasi** — data ditransformasi log (log1p) lalu distandardisasi menggunakan `scaler.pkl`
       yang sama persis dengan yang dipakai saat training model (BAB III).
    3. **Prediksi** — model `kmeans_model.pkl` (K-Means, k=4) memprediksi cluster berdasarkan
       jarak data ke centroid tiap cluster.
    4. **Output** — sistem menerjemahkan nomor cluster menjadi label segmen bisnis
       dan menampilkan rekomendasi strategi terkait.
    """)

    st.subheader("Profil Segmen yang Ditemukan")
    profile_display = pd.DataFrame({
        "Cluster": list(cluster_names.keys()),
        "Nama Segmen": list(cluster_names.values()),
        "Rekomendasi": list(cluster_recs.values())
    })
    st.dataframe(profile_display, use_container_width=True)

    st.caption("Model dilatih menggunakan data transaksi penjualan produk kemasan "
               "periode Agustus 2022 – November 2023 (94 jenis produk).")