import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

from utils import forecast_for_date

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Sistem Rekomendasi Jam Aman Wisata Pantai",
    layout="wide"
)

st.title("🌊 Sistem Rekomendasi Jam Aman Wisata Pantai")
st.caption("Prediksi 24 jam berbasis Random Forest & Knowledge-Based Rules")

# =====================
# LOAD DATA
# =====================
df_model = pd.read_csv("dataset_model.csv", parse_dates=["datetime"])

# =====================
# INPUT
# =====================
selected_date = st.date_input(
    "📅 Pilih Tanggal Prediksi",
    value=pd.Timestamp.today().date()
)

# =====================
# PREDIKSI
# =====================
if st.button("🔮 Prediksi 24 Jam Ke Depan"):

    pred = forecast_for_date(df_model, selected_date)

    # =====================
    # METRIC
    # =====================
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Aman", (pred["Status"] == "Aman").sum())
    c2.metric("🟡 Waspada", (pred["Status"] == "Waspada").sum())
    c3.metric("🔴 Bahaya", (pred["Status"] == "Bahaya").sum())

    # =====================
    # TABLE
    # =====================
    st.subheader("📋 Hasil Prediksi Per Jam")

    def color_status(val):
        if val == "Aman":
            return "background-color:#2ecc71"
        elif val == "Waspada":
            return "background-color:#f1c40f"
        return "background-color:#e74c3c"

    st.dataframe(
        pred.style.applymap(color_status, subset=["Status"]),
        use_container_width=True
    )

    # =====================
    # GRAFIK PASUT
    # =====================
    st.subheader("🌊 Grafik Ketinggian Air")

    chart_pasut = alt.Chart(pred).mark_line(
        point=True,
        strokeWidth=3
    ).encode(
        x=alt.X("Waktu:T", title="Waktu"),
        y=alt.Y("Ketinggian Air (m):Q", title="Ketinggian Air (m)"),
        tooltip=["Waktu", "Ketinggian Air (m)"]
    )

    st.altair_chart(chart_pasut, use_container_width=True)

    # =====================
    # GRAFIK ANGIN
    # =====================
    st.subheader("🌬️ Grafik Kecepatan Angin")

    chart_angin = alt.Chart(pred).mark_line(
        point=True,
        strokeWidth=3
    ).encode(
        x=alt.X("Waktu:T", title="Waktu"),
        y=alt.Y("Kecepatan Angin (km/jam):Q", title="Kecepatan Angin (km/jam)"),
        tooltip=["Waktu", "Kecepatan Angin (km/jam)"]
    )

    st.altair_chart(chart_angin, use_container_width=True)

    # =====================
    # HEATMAP (KECIL)
    # =====================
    st.subheader("🔥 Heatmap Status Keamanan")

    status_map = {"Aman": 0, "Waspada": 1, "Bahaya": 2}
    heat = pred.copy()
    heat["status_num"] = heat["Status"].map(status_map)
    heat["hour"] = heat["Waktu"].dt.hour

    pivot = heat.pivot_table(
        values="status_num",
        index=heat["Waktu"].dt.date,
        columns="hour"
    )

    fig, ax = plt.subplots(figsize=(10, 2.2))
    sns.heatmap(
        pivot,
        cmap=["#2ecc71", "#f1c40f", "#e74c3c"],
        cbar=False,
        linewidths=0.4,
        ax=ax
    )

    ax.set_xlabel("Jam")
    ax.set_ylabel("Tanggal")

    st.pyplot(fig)

    st.info(
        "ℹ️ Data ditampilkan hanya untuk tanggal yang dipilih. "
        "Grafik bersifat interaktif (zoom & hover)."
    )
