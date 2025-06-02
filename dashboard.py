import streamlit as st
import pandas as pd
import plotly.express as px

# ==============================
# Load Data
# ==============================
df = pd.read_excel("/Users/asyzyni/Documents/GitHub/streamlit-warehouse-website/warehouse data.xlsx")

# Format Tanggal
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")

# Format Kolom Persen dan Multiplier
if "Safety Percentage" in df.columns and df["Safety Percentage"].dtype == "object":
    df["Safety Percentage"] = df["Safety Percentage"].str.replace('%', '').astype(float)

df["Event_Multiplier"] = df["Event_Multiplier"].astype(str)

# ==============================
# Konfigurasi Streamlit
# ==============================
st.set_page_config(page_title="Dashboard Permintaan Produk", layout="wide")
st.title("📊 Dashboard Permintaan Produk")

# ==============================
# Sidebar Dropdown (Selectbox)
# ==============================
with st.sidebar:
    st.header("🔍 Pilihan")
    selected_location = st.selectbox("Pilih Lokasi", options=sorted(df["Location"].unique()))
    selected_category = st.selectbox("Pilih Kategori", options=sorted(df["Category"].unique()))

# ==============================
# Filter Data (tanpa filter Event)
# ==============================
filtered_df = df[
    (df["Location"] == selected_location) &
    (df["Category"] == selected_category)
]

# ==============================
# KPI Cards
# ==============================
st.subheader("📌 Ringkasan Kinerja")
col1, col2, col3 = st.columns(3)
col1.metric("Total Permintaan", f"{filtered_df['Demand'].sum():,}")
col2.metric("Rata-rata Permintaan", f"{filtered_df['Demand'].mean():.2f}")
top_product = filtered_df.loc[filtered_df["Demand"].idxmax(), "Type"] if not filtered_df.empty else "-"
col3.metric("Produk Terpopuler", top_product)

# ==============================
# Grafik Tren Permintaan
# ==============================
st.subheader("📈 Tren Permintaan")
fig_trend = px.line(
    filtered_df.sort_values("Date"),
    x="Date", y="Demand",
    color="Category",
    title="Tren Permintaan per Hari"
)
st.plotly_chart(fig_trend, use_container_width=True)

# ==============================
# Grafik Permintaan per Lokasi
# ==============================
st.subheader("📍 Permintaan per Lokasi")
fig_location = px.bar(
    filtered_df.groupby("Location")["Demand"].sum().reset_index(),
    x="Location", y="Demand",
    title="Total Permintaan Berdasarkan Lokasi",
    text_auto=True
)
st.plotly_chart(fig_location, use_container_width=True)

# ==============================
# Pie Chart Kategori
# ==============================
# ==============================
# Pie Chart: Jumlah Type Unik per Kategori
# ==============================
st.subheader("🎯 Distribusi Jumlah Produk (Type) Unik per Kategori")

type_count_per_category = (
    filtered_df.groupby("Category")["Type"]
    .nunique()
    .reset_index()
    .rename(columns={"Type": "Unique_Types"})
)

fig_pie = px.pie(
    type_count_per_category,
    names="Category",
    values="Unique_Types",
    title="Jumlah Produk Unik (Type) per Kategori"
)
st.plotly_chart(fig_pie, use_container_width=True)

# ==============================
# Grafik Demand vs Moving Average
# ==============================
st.subheader("📉 Demand vs Moving Average")
fig_compare = px.line(
    filtered_df.sort_values("Date"),
    x="Date", y=["Demand", "Moving_Average"],
    title="Demand vs Moving Average"
)
st.plotly_chart(fig_compare, use_container_width=True)

# ==============================
# Tabel Detail
# ==============================
st.subheader("📋 Tabel Data Lengkap")
st.dataframe(filtered_df.sort_values("Date"), use_container_width=True)
