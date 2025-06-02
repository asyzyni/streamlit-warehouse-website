import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, date, timedelta
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from typing import Dict
import json
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

class WarehouseDemandApp:
    def __init__(self):
        # Mendapatkan path absolut ke folder artifacts

        
        # Load model dan file pendukung dengan path absolut
        self.model = joblib.load("xgboost_model.pkl")
        self.encoder = joblib.load("onehot_encoder.pkl")
        self.scaler = joblib.load("standard_scaler.pkl")

        # Load column mapping
        with open(os.path.join("column_mapping.json"), "r") as f:
            column_mapping = json.load(f)
            self.categorical_cols = column_mapping["categorical_cols"]
            self.numerical_cols = column_mapping["numerical_cols"]

        # Load data historis
        data_path =  "warehouse data.xlsx"
        self.historical_df = pd.read_excel(data_path)
        self.historical_df['Date'] = pd.to_datetime(self.historical_df['Date'])
                                                    
        self.LOCATIONS = ["Batam", "Bekasi", "Cibitung", "Jakarta", "Makassar", "Surabaya", "Medan", "Tangerang", "Semarang", "Sidoarjo"]
        self.PRODUCT_CATALOG = {
            "AC (Air Conditioner)": ["Cassette 2 PK", "Portable AC 1 PK", "Split 0.5 PK", "Split 1 PK", "Split 1.5 PK Inverter", "Standing Floor 2.5 PK"],
            "Televisi": ["LED 24'", "LED 32'", "LED 43'", "Smart TV 50'", "OLED 55'", "QLED 65'"],
            "Kulkas": ["1 Pintu", "2 Pintu", "Side-by-side", "Mini", "Showcase", "Inverter 2 Pintu"],
            "Mesin Cuci": ["Top Load 6KG", "Top Load 8KG", "Front Load 7KG", "Front Load 10kg", "Twin Tub 9kg", "Inverter 9kg"],
            "Setrika": ["Setrika"],
            "Dispenser": ["Top Loading", "Bottom Loading (Galon Bawah)", "Hot & Cool", "Dispenser Portable"],
            "Microwave": ["Solo Microwave", "Grill Microwave", "Digital Microwave", "Inverter Microwave"],
            "Blender": ["Blender Kaca", "Blender Plastik", "Mini Blender", "2 in 1 Blender (Blender + Grinder)"],
            "Kipas Angin": ["Kipas Meja", "Kipas Berdiri", "Kipas Dinding", "Kipas Remote Control"],
        }
        self.LOCATIONS_COORDINATES = {
            "Batam": (-1.0489, 104.0305), "Bekasi": (-6.2383, 106.9756), "Cibitung": (-6.2684, 107.0844),
            "Jakarta": (-6.2088, 106.8456), "Makassar": (-5.1477, 119.4327), "Surabaya": (-7.2504, 112.7688),
            "Medan": (3.5952, 98.6722), "Tangerang": (-6.1783, 106.6319), "Semarang": (-7.0051, 110.4381), "Sidoarjo": (-7.4478, 112.7183)
        }


    def detect_event(input_date: date):
        events = {
            (1, 1): ("New_Year", 1.2, 0.3), (8, 17): ("Kemerdekaan", 1.2, 0.3),
            (12, 25): ("Natal", 1.4, 0.4), (1, 29): ("Imlek", 1.3, 0.35),
            (3, 29): ("Nyepi", 1.1, 0.25), (4, 2): ("Idul_Fitri", 1.5, 0.45), (6, 7): ("Idul_Adha", 1.3, 0.35)
        }
        if input_date.month == input_date.day:
            return ("Tanggal_Kembar", 1.3, 0.35)
        for (month, day), (event_name, factor1, factor2) in events.items():
            event_date = date(input_date.year, month, day)
            if abs((input_date - event_date).days) <= 7:
                return (event_name, factor1, factor2)
        return ("No_Event", 1.0, 0.15)

    def get_latest_moving_average(self, cat, typ, input_date):
        mask = (
            (self.historical_df["Category"] == cat) &
            (self.historical_df["Type"] == typ) &
            (self.historical_df["Date"] < pd.to_datetime(input_date))
        )
        filtered = self.historical_df[mask].sort_values("Date", ascending=False)
        return filtered.iloc[0]["Moving_Average"] if not filtered.empty else self.historical_df["Moving_Average"].mean()

    def run(self):
        st.title("Prediksi Demand Gudang")
        location = st.selectbox("Pilih Lokasi", self.LOCATIONS)
        category = st.selectbox("Pilih Kategori Produk", list(self.PRODUCT_CATALOG.keys()))
        product_type = st.selectbox("Pilih Tipe Produk", self.PRODUCT_CATALOG[category])
        input_date = st.date_input("Tanggal", min_value=date(2025, 1, 1), max_value=date(2025, 12, 31))

        if st.button("Prediksi Demand"):
            event, multiplier, safety = self.detect_event(input_date)
            moving_avg = self.get_latest_moving_average(category, product_type, input_date)

            data = pd.DataFrame([{
                "Category": category, "Type": product_type, "Location": location,
                "Event": event, "Event_Multiplier": multiplier, "Safety Percentage": safety,
                "Moving_Average": moving_avg, "Multiplier_Safety": multiplier * safety
            }])

            encoded = self.encoder.transform(data[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.categorical_cols))
            scaled = self.scaler.transform(data[self.numerical_cols])
            scaled_df = pd.DataFrame(scaled, columns=self.numerical_cols)
            final_input = pd.concat([scaled_df, encoded_df], axis=1)
            raw_pred = self.model.predict(final_input)[0]
            final_pred = raw_pred * multiplier * (1 + safety)
            st.success(f"Prediksi Demand : {int(round(final_pred))} unit")

        self.show_trend_chart()
        self.show_map()

    def show_trend_chart(self):
        st.markdown("---")
        st.subheader("Tren Pemesanan Tahunan")
        cat = st.selectbox("Pilih Kategori Produk", self.historical_df['Category'].unique())
        mode = st.radio("Pilih Mode Tampilan", ["Single Line Chart", "Multi-Line Chart"])
        df = self.historical_df[self.historical_df['Category'] == cat].copy()
        df['Year'] = df['Date'].dt.year
        df = df[df['Year'].between(2020, 2025)]

        if mode == "Single Line Chart":
            typ = st.selectbox("Pilih Tipe Produk", df['Type'].unique())
            df = df[df['Type'] == typ]
            summary = df.groupby('Year')['Demand'].sum().reset_index()
            fig = px.line(summary, x='Year', y='Demand', markers=True, title=f'Tren Pemesanan Tahunan: {typ}')
        else:
            summary = df.groupby(['Year', 'Type'])['Demand'].sum().reset_index()
            fig = px.line(summary, x='Year', y='Demand', color='Type', markers=True, title=f'Tren Pemesanan Tahunan : {cat}')

        fig.update_layout(xaxis_title='Tahun', yaxis_title='Jumlah Pemesanan', xaxis=dict(dtick=1), yaxis=dict(range=[0, summary['Demand'].max()*1.1]))
        st.plotly_chart(fig)

    def show_map(self):
        st.markdown("---")
        st.subheader("Peta Permintaan Berdasarkan Lokasi")
        df = self.historical_df.copy()
        coord_df = pd.DataFrame.from_dict(self.LOCATIONS_COORDINATES, orient='index', columns=['lat', 'lon']).reset_index().rename(columns={'index': 'Location'})
        df = df.merge(coord_df, on='Location', how='left')
        category_options = ["Semua Kategori"] + sorted(df['Category'].unique())
        selected_category = st.selectbox("Pilih Kategori Produk", category_options)
        if selected_category != "Semua Kategori":
            df = df[df['Category'] == selected_category]
        agg = df.groupby(['Location', 'lat', 'lon'])['Demand'].sum().reset_index()
        fig = px.scatter_mapbox(agg, lat='lat', lon='lon', size='Demand', color_discrete_sequence=["#636EFA"], hover_name='Location', hover_data={'Demand': True, 'lat': False, 'lon': False}, zoom=4, height=600)
        fig.update_layout(mapbox_style="carto-positron", title="Peta Total Permintaan per Lokasi", margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig)

if __name__ == "__main__":
    app = WarehouseDemandApp()
    app.run()