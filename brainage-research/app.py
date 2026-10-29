import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from pathlib import Path

st.set_page_config(page_title="BrainAge Research", page_icon="🧠",
                   layout="wide")

st.title("🧠 BrainAge Research Dashboard")
st.caption("Interactive Neuroscience Analytics | Research Use Only")

files = st.sidebar.file_uploader(
    "Upload datasets",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True
)

if not files:
    st.info("Upload one or more CSV or Excel research datasets.")
    st.stop()

datasets = {}

for f in files:
    try:
        if f.name.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(f.getvalue()))
        else:
            df = pd.read_excel(BytesIO(f.getvalue()))
        datasets[f.name] = df
    except Exception as e:
        st.error(f"{f.name}: {e}")

if not datasets:
    st.stop()

name = st.selectbox("Select dataset", list(datasets))
df = datasets[name].copy()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric("Missing Values", int(df.isna().sum().sum()))

data_tab, quality_tab, chart_tab, corr_tab = st.tabs(
    ["📋 Data", "🔍 Quality", "📊 Charts", "🔗 Correlation"]
)

with data_tab:
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(name).stem}_export.csv",
        mime="text/csv"
    )

with quality_tab:
    quality = pd.DataFrame({
        "Type": df.dtypes.astype(str),
        "Missing": df.isna().sum(),
        "Unique": df.nunique()
    })
    st.dataframe(quality, use_container_width=True)

with chart_tab:
    cols = df.select_dtypes(include="number").columns.tolist()
    if cols:
        x = st.selectbox("X-axis", cols)
        y = st.selectbox("Y-axis", cols, index=min(1, len(cols)-1))
        kind = st.radio("Chart", ["Scatter", "Histogram", "Box"],
                        horizontal=True)
        if kind == "Scatter":
            fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
        elif kind == "Histogram":
            fig = px.histogram(df, x=x, title=f"Distribution of {x}")
        else:
            fig = px.box(df, y=y, title=f"Distribution of {y}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns found.")

with corr_tab:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] >= 2:
        fig = px.imshow(
            numeric.corr(), text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, title="Feature Correlation"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("At least two numeric columns are required.")

st.divider()
st.caption(
    "Educational research only. Not a medical diagnostic tool. "
    "Use authorized datasets and protect participant privacy."
)
