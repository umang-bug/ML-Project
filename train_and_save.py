"""
Run this script ONCE locally to train all models and save them.
Command: python train_and_save.py
Make sure Response.csv is in the same folder (or update the path below).
"""

import pandas as pd
import numpy as np
import joblib
import os

os.makedirs("models", exist_ok=True)

# ──────────────────────────────────────────────
# 1. LOAD & CLEAN RAW DATA
# ──────────────────────────────────────────────
df = pd.read_csv("Response.csv")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Timestamp"]).drop(columns=["Timestamp"])

# Rename helpers
rename_map = {
    "What best describes the place you grew up in?": "Place_Grew_Up",
    "When considering a purchase, how important are the following factors to you?  [Price/Cost]": "Price_Importance",
    "When considering a purchase, how important are the following factors to you?  [Brand reputation]": "Brand_Importance",
    "When considering a purchase, how important are the following factors to you?  [Peer recommendation]": "Peer_Importance",
    "When considering a purchase, how important are the following factors to you?  [Long-term utility/value]": "Utility_Importance",
    "How do you track your monthly expenditures?": "Track_Expenditures",
    "What is the expected graph for your expenditure for the months": "Expenditure_Graph",
    "In which of the following scenarios would you justify an unexpected expense of ₹1,500 or more?": "Justify_Unexpected_Expense",
    "How often do you make purchases that you hadn’t planned for?": "Unplanned_Purchases",
    "On average, how much do you spend per month (excluding tuition fees)?": "Monthly_Spend",
    "On a scale of 1 to 5, how much do social events or peer pressure influence your spending?": "Peer_Influence",
    "How confident do you feel in your ability to manage your personal finances?": "Finance_Confidence",
    "In which categories do you spend the majority of your budget?  [Food & Dining]": "Budget_FoodDining",
    "In which categories do you spend the majority of your budget?  [Travel]": "Budget_Travel",
    "In which categories do you spend the majority of your budget?  [Fashion]": "Budget_Fashion",
    "In which categories do you spend the majority of your budget?  [Subscriptions (Netflix, Spotify, etc.)]": "Budget_Subscriptions",
    "In which categories do you spend the majority of your budget?  [Fun & Entertainment]": "Budget_Entertainment",
}
df.rename(columns=rename_map, inplace=True)
#df.info()
# Budget binary
budget_cols = ["Budget_FoodDining","Budget_Travel","Budget_Fashion","Budget_Subscriptions","Budget_Entertainment"]
for col in budget_cols:
    df[col] = df[col].apply(lambda x: 1 if pd.notna(x) and "Answer" in str(x) else 0)

# Unexpected expense count
df["Justify_Unexpected_Expense"] = df["Justify_Unexpected_Expense"].fillna("")
df["Num_Unexpected_Justifications"] = df["Justify_Unexpected_Expense"].apply(
    lambda x: len(x.split(",")) if x != "" else 0
)
df.drop(columns=["Justify_Unexpected_Expense"], inplace=True)

# Importance mapping
importance_mapping = {"Not important": 0, "Slightly important": 1, "Very important": 2}
for col in ["Price_Importance","Brand_Importance","Peer_Importance","Utility_Importance"]:
    df[col] = df[col].map(importance_mapping)

# Tracking mapping
tracking_mapping = {
    "I do not keep the track": 0,
    "I check my bank balance occasionally.": 1,
    "I review my history within payment apps (e.g., UPI, Paytm).": 2,
    "I use a dedicated expense-tracking app or spreadsheet.": 3,
}
df["Track_Expenditures"] = df["Track_Expenditures"].map(tracking_mapping).fillna(1)

# Graph mapping
graph_mapping = {
    "Uniform Daily Expenses": 0,
    "Steady Weekdays with High Weekends": 1,
    "Irregular and Random Spending": 2,
    "Spend a lot once and then low spending for rest": 3,
    "None": 1,
}
df["Expenditure_Graph"] = df["Expenditure_Graph"].map(graph_mapping).fillna(1)

# Place grew up — clean emojis then one-hot
df["Place_Grew_Up"] = df["Place_Grew_Up"].str.replace(r"[^\w\s-]", "", regex=True).str.strip()
df = pd.get_dummies(df, columns=["Place_Grew_Up"], drop_first=False, dtype=int)

# Ensure all 4 place columns exist
for place in ["Big metro city", "Medium-sized city", "Small town", "Rural area"]:
    col = f"Place_Grew_Up_{place}"
    if col not in df.columns:
        df[col] = 0

print("✅ Data cleaned. Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ──────────────────────────────────────────────
# 2. SYNTHETIC DATA (GMM) → RISK SCORE PIPELINE
# ──────────────────────────────────────────────
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from scipy.spatial.distance import cdist

# GMM synthetic data
gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
gmm.fit(df)
synthetic_data, _ = gmm.sample(500)
df_syn = pd.DataFrame(synthetic_data, columns=df.columns)

for col in df.columns:
    mn, mx = df[col].min(), df[col].max()
    if set(df[col].unique()).issubset({0, 1}):
        df_syn[col] = (df_syn[col] >= 0.5).astype(int)
    else:
        df_syn[col] = np.round(df_syn[col]).clip(mn, mx).astype(int)

# Scale
scaler_rf = StandardScaler()
scaled = scaler_rf.fit_transform(df_syn)

# KMeans risk score
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_syn["Cluster"] = kmeans.fit_predict(scaled)

centroids = kmeans.cluster_centers_
centroids_df = pd.DataFrame(scaler_rf.inverse_transform(centroids), columns=df_syn.columns.drop("Cluster"))
centroids_df["Risk_Metric"] = (
    centroids_df["Unplanned_Purchases"] +
    centroids_df["Peer_Influence"] +
    centroids_df["Expenditure_Graph"]
) - (
    centroids_df["Finance_Confidence"] +
    centroids_df["Track_Expenditures"]
)

safe_cluster_idx = centroids_df["Risk_Metric"].idxmin()
safe_centroid = centroids[safe_cluster_idx]

distances = cdist(scaled, [safe_centroid], metric="euclidean").flatten()
score_scaler = MinMaxScaler(feature_range=(1, 100))
df_syn["KMeans_Risk_Score"] = score_scaler.fit_transform(distances.reshape(-1, 1)).flatten().round(2)

# Persona labels
bins = [0, 33, 66, 100]
persona_labels = ["Saver", "Normal", "Spender"]
df_syn["Financial_Persona"] = pd.cut(df_syn["KMeans_Risk_Score"], bins=bins, labels=persona_labels, include_lowest=True)

# Train RF
X_rf = df_syn.drop(columns=["Cluster","KMeans_Risk_Score","Financial_Persona"], errors="ignore")
y_rf = df_syn["Financial_Persona"]

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_rf, y_rf)
print("✅ Random Forest trained.")

# Save RF pipeline objects
joblib.dump(scaler_rf, "models/scaler_rf.joblib")
joblib.dump(kmeans, "models/kmeans_model.joblib")
joblib.dump(score_scaler, "models/score_scaler.joblib")
joblib.dump(safe_centroid, "models/safe_centroid.joblib")
joblib.dump(rf_classifier, "models/rf_classifier.joblib")
joblib.dump(df_syn.drop(columns=["Cluster","KMeans_Risk_Score","Financial_Persona"], errors="ignore").columns.tolist(),
            "models/rf_feature_columns.joblib")
print("✅ RF + KMeans models saved.")

# ──────────────────────────────────────────────
# 3. K-PROTOTYPES
# ──────────────────────────────────────────────
from kmodes.kprototypes import KPrototypes

df_kp = pd.read_csv("Response.csv")
df_kp.columns = df_kp.columns.str.strip()
df_kp.rename(columns=rename_map, inplace=True)

# Keep raw strings for categorical, raw ints for numeric
kp_budget_cols = ["Budget_FoodDining","Budget_Travel","Budget_Fashion","Budget_Subscriptions","Budget_Entertainment"]
for col in kp_budget_cols:
    df_kp[col] = df_kp[col].apply(lambda x: "Yes" if pd.notna(x) and "Answer" in str(x) else "No")

df_kp["Expenditure_Graph"] = df_kp["Expenditure_Graph"].astype(str)
df_kp["Justify_Unexpected_Expense"] = df_kp["Justify_Unexpected_Expense"].fillna("None")

# Add group column
def get_group(spend):
    if spend in [1, 2, 3]:
        return "Low Spenders"
    elif spend in [4, 5, 6]:
        return "Medium Spenders"
    else:
        return "High Spenders"
df_kp["Group"] = df_kp["Monthly_Spend"].apply(get_group)

# Drop timestamp
df_kp = df_kp.drop(columns=["Timestamp"], errors="ignore")

X_kp = df_kp.drop(columns=["Monthly_Spend"], errors="ignore").copy()
for col in X_kp.columns:
    if not pd.api.types.is_numeric_dtype(X_kp[col]):
        X_kp[col] = X_kp[col].astype(object).fillna("Unknown")
    else:
        X_kp[col] = X_kp[col].fillna(0).astype(float)

cat_indices = [i for i, col in enumerate(X_kp.columns) if X_kp[col].dtype == "object"]

kp_model = KPrototypes(n_clusters=3, init="Cao", n_init=5, random_state=42)
kp_model.fit_predict(X_kp.values, categorical=cat_indices)
print("✅ K-Prototypes trained.")

joblib.dump(kp_model, "models/kproto_model.joblib")
joblib.dump(cat_indices, "models/kproto_cat_indices.joblib")
joblib.dump(X_kp.columns.tolist(), "models/kproto_feature_columns.joblib")
print("✅ K-Prototypes model saved.")

# ──────────────────────────────────────────────
# 4. SOFTMAX NEURAL NETWORK (Spend Tier)
# ──────────────────────────────────────────────
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler as MMS
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

df_nn = pd.read_csv("Response.csv")
df_nn.columns = df_nn.columns.str.strip()
df_nn.rename(columns=rename_map, inplace=True)
df_nn = df_nn.dropna(subset=["Monthly_Spend"])

# Ordinal encoding
OE = OrdinalEncoder(categories=[
    ["Big metro city","Medium-sized city","Small town","Rural area"],
    ["Not important","Slightly important","Very important"],
    ["Not important","Slightly important","Very important"],
    ["Not important","Slightly important","Very important"],
    ["Not important","Slightly important","Very important"],
])
df_nn["Place_Grew_Up"] = df_nn["Place_Grew_Up"].str.replace(r"[^\w\s-]","",regex=True).str.strip()
ordinal_data = OE.fit_transform(df_nn[["Place_Grew_Up","Price_Importance","Brand_Importance","Peer_Importance","Utility_Importance"]])

# One-hot encoding
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_data = ohe.fit_transform(df_nn[["Track_Expenditures","Expenditure_Graph"]])

# JUE binary flags
JUE = pd.DataFrame()
JUE["Emergency(JBS)"] = df_nn["Justify_Unexpected_Expense"].str.contains("Emergencies", na=False).astype(int)
JUE["Discounts(JBS)"] = df_nn["Justify_Unexpected_Expense"].str.contains("discount", na=False, case=False).astype(int)
JUE["Party(JBS)"] = df_nn["Justify_Unexpected_Expense"].str.contains("celebrations", na=False).astype(int)
JUE["Workshop(JBS)"] = df_nn["Justify_Unexpected_Expense"].str.contains("Skill development", na=False).astype(int)
JUE["Trip(JBS)"] = df_nn["Justify_Unexpected_Expense"].str.contains("planned trip", na=False).astype(int)

# Scale numeric
mms = MMS()
numeric_scaled = mms.fit_transform(df_nn[["Unplanned_Purchases","Peer_Influence","Finance_Confidence"]])

# Budget binary
budget_cols_nn = ["Budget_FoodDining","Budget_Travel","Budget_Fashion","Budget_Subscriptions","Budget_Entertainment"]
for col in budget_cols_nn:
    df_nn[col] = df_nn[col].apply(lambda x: 1 if pd.notna(x) and "Answer" in str(x) else 0)

import numpy as np
X_nn = np.hstack([
    ordinal_data,
    ohe_data,
    JUE.values,
    numeric_scaled,
    df_nn[budget_cols_nn].values
])

y_nn = df_nn["Monthly_Spend"].values - 1  # 0-indexed
y_nn_cat = to_categorical(y_nn, num_classes=10)

nn_model = Sequential([
    Dense(60, activation="relu", input_shape=(X_nn.shape[1],)),
    Dense(30, activation="relu"),
    Dense(10, activation="softmax"),
])
nn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
nn_model.fit(X_nn, y_nn_cat, epochs=30, batch_size=20, validation_split=0.15, verbose=0)
print("✅ Neural Network trained.")

nn_model.save("models/spend_nn_model.h5")
joblib.dump(OE, "models/ordinal_encoder.joblib")
joblib.dump(ohe, "models/ohe_encoder.joblib")
joblib.dump(mms, "models/mms_scaler.joblib")
print("✅ Neural Network + encoders saved.")

print("\n🎉 All models saved to /models/. Now push to GitHub and deploy!")
