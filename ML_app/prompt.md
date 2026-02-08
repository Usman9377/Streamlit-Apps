# 🚀 Streamlit AutoML Application Prompt

## Role
Act as an **expert Python Application Developer** specializing in **Streamlit** and **Machine Learning using scikit-learn**.  
Your task is to build a **complete, interactive AutoML web application** with a modern UI, robust preprocessing, multiple ML models, and interactive data visualizations.

---

## 🎯 Application Objective
The app should allow users (with or without ML expertise) to:
- Upload or select datasets
- Explore and visualize data interactively
- Automatically detect ML problem type
- Train and evaluate multiple machine learning models
- Download trained models and predictions
- Make real-time predictions using the trained model

---

## 🖥️ Application Features

### 1️⃣ Welcome & Introduction
- Display a welcoming message with emojis 🤖✨  
- Briefly explain the purpose and workflow of the app.

---

### 2️⃣ Dataset Input (Sidebar)
- Provide a **file uploader** in the sidebar supporting:
  - CSV
  - XLSX
  - TSV
- If no dataset is uploaded, provide a **sample dataset selector** using:
  - `sns.load_dataset()`
  - Available options:
    - `iris`
    - `titanic`
    - `tips`

---

### 3️⃣ Dataset Overview & Exploration
After loading the dataset:
- Display:
  - Dataset shape (rows & columns)
  - Number of missing values
  - Column names
  - Data types
- Show:
  - `head()` of the dataset
  - `describe(include="all")`
- Ensure compatibility with Streamlit display (Arrow-safe DataFrames).

---

### 4️⃣ Interactive Data Visualization 📊
- Use **Plotly** for interactive plots.
- Allow users to:
  - Select X-axis (numeric column)
  - Select Y-axis (numeric column)
  - Select a **color column** (any column)
- Generate interactive scatter plots with:
  - Different colors based on selected column 🎨
  - Zoom, hover, and pan support

---

### 5️⃣ Feature & Target Selection 🎯
- Ask the user to:
  - Select **target variable**
  - Select **feature columns**
- Ensure at least one feature is selected.

---

### 6️⃣ Automatic Problem Detection 🧠
- If target column is:
  - Continuous numeric → **Regression**
  - Categorical or limited unique values → **Classification**
- Display detected problem type clearly.

---

### 7️⃣ Data Preprocessing ⚙️
Implement a robust preprocessing pipeline:
- Handle missing values using:
  - `IterativeImputer`
- Handle categorical variables using:
  - `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
- Keep preprocessing modular using:
  - `ColumnTransformer`
  - `Pipeline`
- Scale numeric features using `StandardScaler`.

---

### 8️⃣ Train-Test Split 🔀
- Allow user to control **test size (%)** via sidebar slider.

---

### 9️⃣ Model Selection 🤖
Provide model selection based on problem type:

#### 🔹 Regression Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor
- Gradient Boosting Regressor

#### 🔹 Classification Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVC)
- K-Nearest Neighbors Classifier
- Naive Bayes
- Gradient Boosting Classifier

---

### 🔟 Model Training & Evaluation 📈

#### Regression Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

#### Classification Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Confusion Matrix (interactive Plotly heatmap)

---

### 1️⃣1️⃣ Model Export 💾
- Allow users to download the trained model as a:
  - `.pkl` (pickle) file

---

### 1️⃣2️⃣ Prediction Interface 🔮
- Ask users if they want to make predictions.
- If yes:
  - Dynamically generate input fields for selected features
  - Predict using the trained pipeline
  - Display predictions clearly

---

### 1️⃣3️⃣ Download Predictions ⬇️
- Allow users to download predictions as:
  - CSV file

---

## 🎨 UI & UX Guidelines
- Use emojis consistently for better engagement 😄
- Use sidebar effectively for controls
- Use wide layout for better visualization
- Ensure app runs without warnings or deprecated arguments
- Compatible with latest `scikit-learn` and `Streamlit` versions

---

## 🚀 Final Output
Deliver a **single `app.py` file** that:
- Runs with `streamlit run app.py`
- Is error-free
- Is production-ready
- Can be deployed on Streamlit Cloud or locally

---

✨ **Bonus Enhancements (Optional)**:
- Model comparison leaderboard
- ROC curves & residual plots
- Hyperparameter tuning
- SHAP-based explainability

---

**End of Prompt** 🎉
