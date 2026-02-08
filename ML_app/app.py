# ==============================
# 🚀 Streamlit AutoML App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle

# Plotting
import plotly.express as px

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==============================
# 🎨 Streamlit Config
# ==============================
st.set_page_config(
    page_title="AutoML Streamlit App",
    page_icon="🤖",
    layout="wide"
)

# ==============================
# 🧠 Helper Functions
# ==============================
def make_arrow_safe(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df


# ==============================
# 👋 App Header
# ==============================
st.title("🤖 AutoML Streamlit App")
st.markdown("""
Welcome to the **Interactive Machine Learning App** 🚀  

✨ Upload your dataset or use a sample  
✨ Automatically detect ML problem type  
✨ Train multiple ML models  
✨ Visualize data interactively  
✨ Download trained models & predictions  
""")

# ==============================
# 📂 Sidebar – Data Input
# ==============================
st.sidebar.header("📂 Data Input Options")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "tsv"]
)

sample_dataset = st.sidebar.selectbox(
    "Or use a sample dataset",
    ["None", "iris", "titanic", "tips"]
)

# ==============================
# 📊 Load Data
# ==============================
df = None

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t")

elif sample_dataset != "None":
    df = sns.load_dataset(sample_dataset)

if df is None:
    st.warning("📌 Please upload a dataset or select a sample dataset")
    st.stop()

# ==============================
# 📋 Dataset Overview
# ==============================
st.subheader("📋 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing Values", df.isnull().sum().sum())

st.dataframe(make_arrow_safe(df.head()))

st.markdown("### 📑 Data Description")
st.dataframe(make_arrow_safe(df.describe(include="all")))

st.markdown("### 🧬 Data Types")
st.write(df.dtypes)

# ==============================
# 📊 Interactive Visualization
# ==============================
st.subheader("📊 Interactive Data Visualization")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

if numeric_cols:
    x_col = st.selectbox("Select X-axis", numeric_cols)
    y_col = st.selectbox("Select Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    color_col = st.selectbox("Color by column 🎨", df.columns)

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title="📈 Interactive Scatter Plot",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# 🎯 Feature & Target Selection
# ==============================
st.subheader("🎯 Feature & Target Selection")

target_col = st.selectbox("Select Target Variable 🎯", df.columns)
feature_cols = st.multiselect(
    "Select Feature Columns 🧩",
    [col for col in df.columns if col != target_col]
)

if not feature_cols:
    st.warning("⚠️ Please select at least one feature")
    st.stop()

X = df[feature_cols]
y = df[target_col]

# ==============================
# 🧠 Problem Type Detection
# ==============================
if y.dtype == "object" or y.nunique() < 15:
    problem_type = "classification"
    st.success("🧠 Detected Problem Type: **Classification**")
else:
    problem_type = "regression"
    st.success("🧠 Detected Problem Type: **Regression**")

# ==============================
# ⚙️ Preprocessing
# ==============================
num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", IterativeImputer()),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", IterativeImputer()),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

# ==============================
# 🔀 Train-Test Split
# ==============================
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# ==============================
# 🤖 Model Selection
# ==============================
st.sidebar.header("🤖 Model Selection")

if problem_type == "regression":
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["Linear Regression", "Decision Tree", "Random Forest", "SVR", "KNN", "Gradient Boosting"]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

else:
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Naive Bayes", "Gradient Boosting"]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

model = models[model_name]

# ==============================
# 🏋️ Train Model
# ==============================
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

# ==============================
# 📈 Evaluation
# ==============================
st.subheader("📈 Model Evaluation")

if problem_type == "regression":
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.metric("MAE", round(mae, 3))
    st.metric("MSE", round(mse, 3))
    st.metric("RMSE", round(rmse, 3))
    st.metric("R² Score", round(r2, 3))

else:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    st.metric("Accuracy", round(acc, 3))
    st.metric("Precision", round(prec, 3))
    st.metric("Recall", round(rec, 3))
    st.metric("F1 Score", round(f1, 3))

    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        title="🧩 Confusion Matrix"
    )
    st.plotly_chart(fig_cm)

# ==============================
# 💾 Download Model
# ==============================
st.subheader("💾 Download Trained Model")

model_bytes = pickle.dumps(pipeline)

st.download_button(
    label="⬇️ Download Model (pickle)",
    data=model_bytes,
    file_name="trained_model.pkl"
)

# ==============================
# 🔮 Prediction Interface
# ==============================
st.subheader("🔮 Make Predictions")

if st.checkbox("Yes, I want to make predictions"):
    input_data = {}
    for col in feature_cols:
        input_data[col] = st.text_input(f"Enter {col}")

    if st.button("Predict 🚀"):
        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)
        st.success(f"🎯 Prediction: {prediction}")

        pred_df = input_df.copy()
        pred_df["Prediction"] = prediction

        st.download_button(
            "⬇️ Download Predictions",
            pred_df.to_csv(index=False),
            "predictions.csv"
        )
