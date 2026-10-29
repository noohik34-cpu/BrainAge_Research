import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="BrainAge Research", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp {background-color:#f5f7fc}
.block-container {padding-top:2rem}
[data-testid="stMetric"] {background:white;padding:18px;border-radius:12px;
border:1px solid #e2e8f0}
</style>
""", unsafe_allow_html=True)

st.title("🧠 BrainAge Research Dashboard")
st.caption("Professional Neuroscience Analytics | Data Science | Machine Learning")

file = st.sidebar.file_uploader("Upload Excel or CSV", type=["xlsx","xls","csv"])

if file is None:
    st.info("👈 Upload your OASIS research dataset to get started.")
    st.stop()

try:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
except Exception as e:
    st.error(f"File error: {e}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()
numeric = df.select_dtypes(include=np.number).columns.tolist()

age = next((c for c in df.columns if c.lower() in
            ["age","actual_age","chronological_age"]), None)

st.sidebar.header("Filters")
if age and pd.to_numeric(df[age], errors="coerce").notna().any():
    df[age] = pd.to_numeric(df[age], errors="coerce")
    valid = df[age].dropna()
    if valid.min() < valid.max():
        limits = st.sidebar.slider("Age range", int(valid.min()),
                                   int(valid.max()),
                                   (int(valid.min()), int(valid.max())))
        data = df[df[age].between(*limits)].copy()
    else:
        data = df.copy()
else:
    data = df.copy()

search = st.sidebar.text_input("Search records")
if search:
    data = data[data.astype(str).apply(
        lambda c: c.str.contains(search, case=False, na=False)
    ).any(axis=1)]

a,b,c,d = st.columns(4)
a.metric("Participants", len(data))
b.metric("Variables", len(data.columns))
c.metric("Missing Values", int(data.isna().sum().sum()))
d.metric("Average Age", f"{data[age].mean():.1f}" if age and data[age].notna().any() else "N/A")

overview, explore, quality, model, download = st.tabs(
    ["📊 Overview","🔍 Explore","🧹 Data Quality","🤖 Age Prediction","📥 Export"]
)

with overview:
    st.subheader("Dataset Preview")
    st.dataframe(data.head(100), width="stretch")
    if numeric:
        col = st.selectbox("Distribution variable", numeric)
        st.plotly_chart(px.histogram(data, x=col, marginal="box",
                                     title=f"Distribution of {col}"),
                        width="stretch")

with explore:
    if len(numeric) >= 2:
        x = st.selectbox("X-axis", numeric)
        y = st.selectbox("Y-axis", [v for v in numeric if v != x])
        st.plotly_chart(px.scatter(data, x=x, y=y, trendline="ols",
                                   title=f"{y} vs {x}"),
                        width="stretch")
        st.plotly_chart(px.imshow(data[numeric].corr(), text_auto=".2f",
                                  color_continuous_scale="RdBu_r",
                                  title="Correlation Heatmap"),
                        width="stretch")
    else:
        st.warning("At least two numeric columns are needed.")

with quality:
    report = pd.DataFrame({
        "Missing": data.isna().sum(),
        "Missing %": (data.isna().mean()*100).round(2)
    })
    st.dataframe(report, width="stretch")
    st.metric("Duplicate Rows", int(data.duplicated().sum()))
    st.plotly_chart(px.bar(report.reset_index(), x="index", y="Missing",
                           title="Missing Values by Column"),
                    width="stretch")

with model:
    features = [v for v in numeric if v != age]
    if age and features:
        selected = st.multiselect("Brain measurements / predictors",
                                  features, default=features[:min(3,len(features))])
        if st.button("🚀 Train Model"):
            if not selected:
                st.warning("Select at least one predictor.")
            else:
                train = data[[age]+selected].dropna(subset=[age]).copy()
                X = train[selected].replace([np.inf,-np.inf],np.nan)
                y = train[age]
                if len(train) < 20 or y.nunique() < 2:
                    st.error("Use at least 20 valid records with varying ages.")
                else:
                    Xtr,Xte,ytr,yte = train_test_split(
                        X,y,test_size=0.25,random_state=42)
                    model = make_pipeline(
                        SimpleImputer(strategy="median"),
                        RandomForestRegressor(n_estimators=100,
                                              random_state=42))
                    model.fit(Xtr,ytr)
                    pred = model.predict(Xte)
                    p1,p2 = st.columns(2)
                    p1.metric("Mean Absolute Error",f"{mean_absolute_error(yte,pred):.2f} years")
                    p2.metric("R² Score",f"{r2_score(yte,pred):.3f}")
                    result = pd.DataFrame({"Actual_Age":yte.values,
                                           "Predicted_Age":pred.round(2)})
                    st.plotly_chart(px.scatter(result,x="Actual_Age",
                                               y="Predicted_Age",
                                               title="Actual vs Predicted Age"),
                                    width="stretch")
                    st.dataframe(result, width="stretch")
                    st.download_button("Download Predictions CSV",
                        result.to_csv(index=False).encode("utf-8-sig"),
                        "brainage_predictions.csv","text/csv")
    else:
        st.warning("A numeric Age column and predictor columns are required.")

with download:
    st.subheader("Export Research Data")
    st.download_button("⬇️ Download Filtered CSV",
        data[data.columns].to_csv(index=False).encode("utf-8-sig"),
        "brainage_data.csv","text/csv")
    st.download_button("⬇️ Download Excel",
        data.to_csv(index=False).encode("utf-8-sig"),
        "brainage_data.csv","text/csv")
    st.write("Download your filtered dataset for further analysis.")

st.divider()
st.caption("BrainAge Research | Educational research only — not a medical diagnostic tool.")