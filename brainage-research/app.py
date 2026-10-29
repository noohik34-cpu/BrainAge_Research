import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score,
    classification_report
)

st.set_page_config(
    page_title="BrainAge Research",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 BrainAge Research Dashboard")
st.caption("Neuroscience Analytics • Interactive Visualization • Machine Learning")

st.sidebar.header("📂 Dataset Upload")
files = st.sidebar.file_uploader(
    "Upload CSV or Excel files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True
)

if not files:
    st.info("Upload a research dataset to explore your data.")
    st.write("Supported formats: CSV, XLSX, XLS.")
    st.caption("Educational research tool — not a medical diagnostic system.")
    st.stop()


def suggest_column(columns, keywords):
    for col in columns:
        name = str(col).lower().replace("_", " ").replace("-", " ")
        if any(word in name for word in keywords):
            return col
    return columns[0] if columns else None


for file in files:
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file, sep=None, engine="python")
        else:
            df = pd.read_excel(file)

        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

        if df.empty or len(df.columns) == 0:
            st.warning(f"{file.name}: Dataset is empty.")
            continue

        st.divider()
        st.header(f"📊 {file.name}")

        # 1. DATA QUALITY AND OVERVIEW
        a, b, c, d = st.columns(4)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", len(df.columns))
        c.metric("Missing cells", f"{df.isna().sum().sum():,}")
        d.metric("Duplicate rows", f"{df.duplicated().sum():,}")

        tabs = st.tabs([
            "📋 Overview", "🔍 Filters & Mapping",
            "📈 Interactive Charts", "🤖 Machine Learning"
        ])

        with tabs[0]:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(100), use_container_width=True)

            st.subheader("Data Quality")
            quality = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing": df.isna().sum().values,
                "Unique Values": df.nunique().values
            })
            st.dataframe(quality, use_container_width=True)

            numeric = df.select_dtypes(include=np.number)
            if not numeric.empty:
                st.subheader("Statistical Summary")
                st.dataframe(numeric.describe().T, use_container_width=True)

            st.download_button(
                "⬇️ Download Dataset",
                df.to_csv(index=False).encode("utf-8"),
                file_name="research_dataset.csv",
                mime="text/csv",
                key=f"download_{files.index(file)}"
            )

        with tabs[1]:
            st.subheader("Automatic Column Mapping")

            cols = df.columns.tolist()
            age_guess = suggest_column(
                cols, ["age", "age years", "participant age"]
            )
            gender_guess = suggest_column(
                cols, ["gender", "sex"]
            )
            volume_guess = suggest_column(
                cols, ["brain volume", "brainvol", "icv", "volume"]
            )
            mmse_guess = suggest_column(
                cols, ["mmse", "cognitive score", "cognition"]
            )

            def mapped_select(label, guess, key):
                options = ["Not available"] + cols
                default = options.index(guess) if guess in cols else 0
                return st.selectbox(
                    label, options, index=default, key=key
                )

            age_col = mapped_select(
                "Age column", age_guess, f"age_{files.index(file)}"
            )
            gender_col = mapped_select(
                "Gender / Sex column", gender_guess,
                f"gender_{files.index(file)}"
            )
            volume_col = mapped_select(
                "Brain volume column", volume_guess,
                f"volume_{files.index(file)}"
            )
            mmse_col = mapped_select(
                "MMSE / Cognitive score column", mmse_guess,
                f"mmse_{files.index(file)}"
            )

            st.caption(
                "Automatic mapping is a suggestion. Verify each column "
                "and its units before interpreting results."
            )

            filtered = df.copy()
            filter_cols = st.multiselect(
                "Choose columns to filter",
                cols,
                key=f"filters_{files.index(file)}"
            )

            for col in filter_cols:
                if pd.api.types.is_numeric_dtype(filtered[col]):
                    values = pd.to_numeric(
                        filtered[col], errors="coerce"
                    ).dropna()

                    if not values.empty and values.min() < values.max():
                        low, high = st.slider(
                            f"Filter {col}",
                            float(values.min()),
                            float(values.max()),
                            (float(values.min()), float(values.max())),
                            key=f"slider_{files.index(file)}_{col}"
                        )
                        filtered = filtered[
                            filtered[col].isna()
                            | filtered[col].between(low, high)
                        ]
                else:
                    values = filtered[col].dropna().astype(str).unique().tolist()
                    selected = st.multiselect(
                        f"Select {col}",
                        values,
                        default=values,
                        key=f"category_{files.index(file)}_{col}"
                    )
                    filtered = filtered[
                        filtered[col].isna()
                        | filtered[col].astype(str).isin(selected)
                    ]

            st.success(f"{len(filtered):,} rows after filtering")
            st.dataframe(filtered.head(100), use_container_width=True)

            st.download_button(
                "⬇️ Download Filtered Data",
                filtered.to_csv(index=False).encode("utf-8"),
                file_name="filtered_research_data.csv",
                mime="text/csv",
                key=f"filtered_{files.index(file)}"
            )

            # Save selected mappings for the chart tab.
            st.session_state[f"mapping_{files.index(file)}"] = {
                "Age": age_col,
                "Gender": gender_col,
                "Brain Volume": volume_col,
                "MMSE": mmse_col
            }
            st.session_state[f"filtered_{files.index(file)}"] = filtered

        with tabs[2]:
            st.subheader("Interactive Data Visualization")

            chart_data = st.session_state.get(
                f"filtered_{files.index(file)}", df
            )
            numeric_cols = chart_data.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_cols:
                x = st.selectbox(
                    "X-axis", numeric_cols,
                    key=f"chart_x_{files.index(file)}"
                )
                y_options = [col for col in numeric_cols if col != x]

                if y_options:
                    y = st.selectbox(
                        "Y-axis", y_options,
                        key=f"chart_y_{files.index(file)}"
                    )
                    chart_type = st.selectbox(
                        "Chart type",
                        ["Scatter", "Line", "Histogram", "Box"],
                        key=f"chart_type_{files.index(file)}"
                    )

                    if chart_type == "Scatter":
                        fig = px.scatter(
                            chart_data, x=x, y=y,
                            trendline=None,
                            title=f"{y} vs {x}"
                        )
                    elif chart_type == "Line":
                        fig = px.line(chart_data, x=x, y=y)
                    elif chart_type == "Histogram":
                        fig = px.histogram(chart_data, x=x)
                    else:
                        fig = px.box(chart_data, x=x, y=y)

                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Correlation Matrix")
                corr = chart_data[numeric_cols].corr()
                fig = px.imshow(
                    corr, text_auto=".2f",
                    aspect="auto", title="Numeric Feature Correlations"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for these charts.")

            mapping = st.session_state.get(
                f"mapping_{files.index(file)}", {}
            )
            age = mapping.get("Age")
            volume = mapping.get("Brain Volume")

            if (
                age in chart_data.columns
                and volume in chart_data.columns
                and age != "Not available"
                and volume != "Not available"
            ):
                plot_data = chart_data.copy()
                plot_data[age] = pd.to_numeric(
                    plot_data[age], errors="coerce"
                )
                plot_data[volume] = pd.to_numeric(
                    plot_data[volume], errors="coerce"
                )
                plot_data = plot_data.dropna(subset=[age, volume])

                if not plot_data.empty:
                    st.subheader("Age vs Brain Volume")
                    fig = px.scatter(
                        plot_data, x=age, y=volume,
                        trendline=None,
                        title="Age and Brain Volume"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            st.subheader("Train a Machine-Learning Model")

            st.warning(
                "Predictions depend on the uploaded data and selected target. "
                "This is not a clinically validated BrainAge model."
            )

            ml_data = st.session_state.get(
                f"filtered_{files.index(file)}", df
            )
            target = st.selectbox(
                "Select prediction target",
                ml_data.columns.tolist(),
                key=f"target_{files.index(file)}"
            )

            task = st.radio(
                "Prediction task",
                ["Regression (numeric target)", "Classification (categories)"],
                key=f"task_{files.index(file)}"
            )

            if st.button(
                "🚀 Train Model",
                key=f"train_{files.index(file)}"
            ):
                try:
                    data_ml = ml_data.dropna(subset=[target]).copy()
                    X = data_ml.drop(columns=[target])
                    y = data_ml[target]

                    if len(data_ml) < 20:
                        st.error("Use at least 20 labelled rows for this demo.")
                    elif X.shape[1] == 0:
                        st.error("At least one input feature is required.")
                    else:
                        # Remove constant or entirely missing features.
                        X = X.loc[:, X.nunique(dropna=True) > 1]
                        X = X.dropna(axis=1, how="all")

                        if X.shape[1] == 0:
                            st.error("No usable input features were found.")
                            continue

                        # Exclude obvious identifiers from model inputs.
                        id_cols = [
                            col for col in X.columns
                            if str(col).lower() in [
                                "id", "subject id", "participant id",
                                "record id", "patient id"
                            ]
                        ]
                        X = X.drop(columns=id_cols)

                        if X.shape[1] == 0:
                            st.error("No usable features remain after removing IDs.")
                            continue

                        if task.startswith("Regression"):
                            y = pd.to_numeric(y, errors="coerce")
                            valid = y.notna()
                            X, y = X.loc[valid], y.loc[valid]
                            model = RandomForestRegressor(
                                n_estimators=150, random_state=42,
                                n_jobs=-1
                            )
                        else:
                            y = y.astype(str)
                            if y.nunique() < 2:
                                st.error("Classification needs at least 2 classes.")
                                continue
                            model = RandomForestClassifier(
                                n_estimators=150, random_state=42,
                                n_jobs=-1, class_weight="balanced"
                            )

                        numeric_features = X.select_dtypes(
                            include=np.number
                        ).columns.tolist()
                        categorical_features = [
                            col for col in X.columns
                            if col not in numeric_features
                        ]

                        transformers = []
                        if numeric_features:
                            transformers.append((
                                "num",
                                SimpleImputer(strategy="median"),
                                numeric_features
                            ))
                        if categorical_features:
                            transformers.append((
                                "cat",
                                Pipeline([
                                    ("imputer", SimpleImputer(
                                        strategy="most_frequent"
                                    )),
                                    ("encoder", OneHotEncoder(
                                        handle_unknown="ignore"
                                    ))
                                ]),
                                categorical_features
                            ))

                        preprocess = ColumnTransformer(
                            transformers=transformers
                        )
                        pipeline = Pipeline([
                            ("preprocess", preprocess),
                            ("model", model)
                        ])

                        stratify = None
                        if task.startswith("Classification"):
                            counts = y.value_counts()
                            if len(counts) >= 2 and counts.min() >= 2:
                                stratify = y

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2,
                            random_state=42, stratify=stratify
                        )

                        pipeline.fit(X_train, y_train)
                        predictions = pipeline.predict(X_test)

                        st.success("Model training completed.")

                        if task.startswith("Regression"):
                            m1, m2 = st.columns(2)
                            m1.metric(
                                "Mean Absolute Error",
                                f"{mean_absolute_error(y_test, predictions):.3f}"
                            )
                            m2.metric(
                                "R² Score",
                                f"{r2_score(y_test, predictions):.3f}"
                            )
                        else:
                            st.metric(
                                "Test Accuracy",
                                f"{accuracy_score(y_test, predictions):.1%}"
                            )
                            st.text(classification_report(
                                y_test, predictions, zero_division=0
                            ))

                        results = pd.DataFrame({
                            "Actual": y_test.to_numpy(),
                            "Predicted": predictions
                        })
                        st.subheader("Test Set Predictions")
                        st.dataframe(results, use_container_width=True)

                        st.download_button(
                            "⬇️ Download Test Predictions",
                            results.to_csv(index=False).encode("utf-8"),
                            file_name="model_test_predictions.csv",
                            mime="text/csv",
                            key=f"predictions_{files.index(file)}"
                        )

                        st.caption(
                            "Metrics are based on a held-out test split. "
                            "They do not establish medical validity or "
                            "guarantee performance on other datasets."
                        )

                except Exception as e:
                    st.error(f"Model training failed: {e}")

    except Exception as e:
        st.error(f"Could not read {file.name}: {e}")

st.divider()
st.caption(
    "BrainAge Research • Educational research only • "
    "Protect patient privacy and use only authorized data."
)
