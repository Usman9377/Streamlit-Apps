import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
# streamlit_env environment made for this
# ===================== APP HEADER =====================
st.set_page_config(page_title="📊 Data Analysis App", layout="wide")

st.title("📊 Interactive Data Analysis App")
st.subheader("🚀 A simple yet powerful data analysis app by **Muhammad Usman**")

st.markdown("---")

# ===================== DATASET SELECTION =====================
st.header("📂 Dataset Selection")

selected_dataset = st.selectbox(
    'Choose a dataset 👇',
    ['iris', 'tips', 'titanic', 'diamonds', 'Upload your own']
)

if selected_dataset == 'Upload your own':
    uploaded_file = st.file_uploader('📤 Upload your CSV file', type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success('✅ File uploaded successfully!')
    else:
        st.info('ℹ️ Please upload a CSV file to continue.')
        df = None
else:
    df = sns.load_dataset(selected_dataset)

# ===================== DATA PREVIEW =====================
if df is not None:
    st.header("👀 Dataset Preview")
    st.dataframe(df)

    col1, col2 = st.columns(2)
    col1.metric("📏 Rows", df.shape[0])
    col2.metric("📐 Columns", df.shape[1])

    st.subheader("🧾 Column Names & Data Types")
    st.write(df.dtypes)

# ===================== MISSING VALUES =====================
if df is not None:
    st.subheader("🧼 Missing Values Check")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]

    if not null_counts.empty:
        st.warning("⚠️ Columns with missing values")
        st.write(null_counts)
    else:
        st.success("✅ No missing values found!")

# ===================== VISUALIZATION SECTION =====================
if df is not None:

    st.markdown("---")
    st.header("📊 Interactive Data Visualization")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # 🎨 Color Theme Selector
    color_theme = st.selectbox(
        "🎨 Select Color Theme",
        ["plotly", "ggplot2", "seaborn", "simple_white", "plotly_dark"]
    )

    plot_type = st.selectbox(
        "📈 Select Plot Type",
        [
            "Scatter Plot",
            "Line Plot",
            "Histogram",
            "Box Plot",
            "Bar Plot",
            "Area Plot",
            "Violin Plot",
            "Correlation Heatmap"
        ]
    )

    # ===================== Scatter & Line =====================
    if plot_type in ["Scatter Plot", "Line Plot"]:
        x_col = st.selectbox("➡️ X-axis", numeric_cols)
        y_col = st.selectbox("⬆️ Y-axis", numeric_cols, index=1)
        color_col = st.selectbox("🎯 Group / Color (optional)", [None] + categorical_cols)

        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col,
            template=color_theme,
            title=f"📌 {plot_type}: {x_col} vs {y_col}"
        ) if plot_type == "Scatter Plot" else px.line(
            df, x=x_col, y=y_col, color=color_col,
            template=color_theme,
            title=f"📌 {plot_type}: {x_col} vs {y_col}"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ===================== Histogram =====================
    elif plot_type == "Histogram":
        x_col = st.selectbox("📊 Select Numeric Column", numeric_cols)

        fig = px.histogram(
            df, x=x_col, nbins=30,
            template=color_theme,
            title=f"📊 Histogram of {x_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===================== Box Plot =====================
    elif plot_type == "Box Plot":
        y_col = st.selectbox("📦 Y-axis (Numeric)", numeric_cols)
        x_col = st.selectbox("🎯 Group By (optional)", [None] + categorical_cols)

        fig = px.box(
            df, x=x_col, y=y_col,
            template=color_theme,
            title=f"📦 Box Plot of {y_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===================== Bar Plot =====================
    elif plot_type == "Bar Plot":
        x_col = st.selectbox("📊 Category Column", categorical_cols)
        y_col = st.selectbox("📏 Numeric Column", numeric_cols)

        fig = px.bar(
            df, x=x_col, y=y_col,
            template=color_theme,
            title=f"📊 Bar Plot: {y_col} by {x_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===================== Area Plot =====================
    elif plot_type == "Area Plot":
        x_col = st.selectbox("➡️ X-axis", numeric_cols)
        y_col = st.selectbox("⬆️ Y-axis", numeric_cols, index=1)

        fig = px.area(
            df, x=x_col, y=y_col,
            template=color_theme,
            title=f"🌊 Area Plot: {x_col} vs {y_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===================== Violin Plot =====================
    elif plot_type == "Violin Plot":
        y_col = st.selectbox("🎻 Y-axis (Numeric)", numeric_cols)
        x_col = st.selectbox("🎯 Category (optional)", [None] + categorical_cols)

        fig = px.violin(
            df, x=x_col, y=y_col,
            box=True, points="all",
            template=color_theme,
            title=f"🎻 Violin Plot of {y_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===================== Correlation Heatmap =====================
    elif plot_type == "Correlation Heatmap":
        st.subheader("🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("🔥 Correlation Heatmap")

        st.pyplot(fig)
