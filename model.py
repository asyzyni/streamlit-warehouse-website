import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import xgboost as xgb
import pickle
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor
from category_encoders import TargetEncoder
import os
from xgboost import plot_importance
from category_encoders import MEstimateEncoder
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

df = pd.read_excel("/content/drive/MyDrive/TUBES PASD/warehouse data.xlsx")

# Feature Engineering

# 1. Time Based Features
df['Date'] = pd.to_datetime(df['Date'])
df['Day_of_week'] = df['Date'].dt.dayofweek
df['Is_weekend'] = df['Day_of_week'].isin([5, 6]).astype(int)

# 2. Binary Feature
df['Is_Event'] = df['Event'].apply(lambda x: 1 if x != 'No_Event' else 0)

# 3. Interaksi Fitur
df['Multiplier_Safety'] = df['Event_Multiplier'] * df['Safety Percentage']

# Kolom kategori untuk One-Hot Encoding
categorical_cols = ['Category', 'Type', 'Location', 'Event']

# Kolom numerik untuk scaling
numerical_cols = ['Event_Multiplier', 'Safety Percentage', 'Moving_Average',
                  'Multiplier_Safety']

# One-Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

# Scaling untuk numerik
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(df[numerical_cols])
scaled_df = pd.DataFrame(scaled_numerical, columns=numerical_cols)


# Gabungkan semua fitur
X = pd.concat([scaled_df, encoded_df], axis=1)
y = df['Demand']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Inisialisasi model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    early_stopping_rounds=50,
    eval_metric='rmse'
)

# Training dengan early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=10
)

# Evaluasi
score = model.score(X_test, y_test)
print(f"R-squared: {score:.4f}")

# Simpan model dan preprocessing objects
joblib.dump(model, 'xgboost_model.pkl')

# Simpan encoder per fitur (opsional, jika ingin fleksibel di Streamlit)
joblib.dump(encoder, 'onehot_encoder.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')

# Simpan daftar kolom untuk referensi
import json
with open('column_mapping.json', 'w') as f:
    json.dump({
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }, f)

