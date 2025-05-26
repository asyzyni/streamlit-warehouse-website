import streamlit as st
import pandas as pd
import joblib 
import pickle as pkl 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

class DemandPredictor():
    def __init__(self):
        self.model = xgb.XGBRegressor(),
        self.encoders = {
            'label-category' : None,
            'label-location' : None, 
            'label-event' : None,
            'label-type' : None
        }
        self.load_artifacts()
        
    def load_artifacts(self):
        try:
            self.model.load_model("/Users/asyzyni/Documents/GitHub/streamlit-warehouse-website/best_xgboost_demand_predictior.model")
            self.encoders['label-category'] = joblib.load("/Users/asyzyni/Documents/GitHub/streamlit-warehouse-website/Encoder/label_encoder_Category.joblib")
            self.encoders['label-event'] = joblib.load("/Users/asyzyni/Documents/GitHub/streamlit-warehouse-website/Encoder/label_encoder_Event.joblib")
            self.encoders['label-location'] = joblib.load("/Users/asyzyni/Documents/GitHub/streamlit-warehouse-website/Encoder/label_encoder_Location.joblib")
            self.encoders['label-type'] = joblib.load("/Users/asyzyni/Documents/GitHub/streamlit-warehouse-website/Encoder/label_encoder_Type.joblib")
        
        except Exception as e:
            st.error(f"Cant load artifacts : {e}")
            st.stop()
            
    # baru sampai load artifacts 
    # belum urus perubahan yang nanti di tanggal 
    # belum urus feature engineering yang lain. 
    # bye selamat bobo 
    