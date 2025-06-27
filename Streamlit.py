import streamlit as st
import joblib
import os
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Energy Efficiency Classifier", layout="wide")


# # Debugging: Show current working directory and contents of images folder
# st.write("📂 Current working directory:", os.getcwd())
# try:
#     st.write("🖼️ Images folder contents:", os.listdir("images"))
# except FileNotFoundError:
#     st.error("❌ 'images/' folder not found. Make sure it's in the same directory as this script.")

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("best_random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ========== STYLING ==========
st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    padding: 0.6rem 1.2rem;
    font-size: 1rem;
    font-weight: 600;
    background-color: #393E46;
    color: white;
    border: none;
    border-radius: 10px;
    transition: 0.2s ease;
}
div.stButton > button:hover {
    background-color: #4e555e;
}
</style>
""", unsafe_allow_html=True)

# ========== TITLE ==========
st.markdown("<h2 style='text-align:center;'>🔍 Energy Efficiency Classifier</h2>", unsafe_allow_html=True)

# ========== NAVIGATION ==========
nav_cols = st.columns([2, 2, 2, 2, 2])  # center 3 buttons

with nav_cols[1]:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"
with nav_cols[2]:
    if st.button("🧪 Try Model"):
        st.session_state.page = "Try"
with nav_cols[3]:
    if st.button("🛠️ Model Development"):
        st.session_state.page = "Dev"

st.markdown("---")

# ========== HOME ==========
if st.session_state.page == "Home":
    st.header("🏠 Welcome to the Energy Efficiency Classifier")
    st.markdown("""
    This tool classifies the energy efficiency of buildings using a fine-tuned Random Forest model.

    **Steps:**
    - Head over to **Try Model** to input your building's specifications.
    - View how the classifier predicts heating load efficiency.

    Dataset source: UCI Energy Efficiency.
    """)

    st.markdown("### 📌 Project Objective")
    st.markdown("""
    - Develop a machine learning model that predicts the energy efficiency class of a building based on its physical characteristics.
    - Assist designers and engineers in optimizing energy consumption during the early design stages.
    - Promote sustainable architecture through intelligent, data-driven design decisions.
    - Provide a user-friendly web interface to evaluate building designs in real time.
    """)

    st.markdown("### 🌍 Sustainable Development Goals (SDG) Alignment")
    st.markdown("""
    This project supports the following United Nations SDGs:

    - **🏡 SDG 7: Affordable and Clean Energy**  
      By promoting energy-efficient building designs, the project reduces dependency on non-renewable energy.

    - **🏗️ SDG 9: Industry, Innovation and Infrastructure**  
      Encourages the use of innovative technologies (AI/ML) in construction and design sectors.

    - **🌱 SDG 11: Sustainable Cities and Communities**  
      Supports eco-friendly building design, contributing to more sustainable urban environments.

    - **♻️ SDG 13: Climate Action**  
      Reducing heating load through better building designs helps lower overall carbon emissions.
    """)

# ========== TRY MODEL ==========
elif st.session_state.page == "Try":
    st.header("🧪 Try the Energy Efficiency Model")
    with st.form("classifier_form"):
        st.markdown("### Input Building Characteristics")
        col1, col2 = st.columns(2)

        with col1:
            x1 = st.number_input("Relative Compactness (X1)", 0.62, 0.98, 0.76)
            x2 = st.number_input("Surface Area (X2)", 514.5, 808.5, 784.0)
            x3 = st.number_input("Wall Area (X3)", 245.0, 416.5, 343.0)
            x4 = st.number_input("Roof Area (X4)", 110.25, 441.0, 147.0)

        with col2:
            x5 = 3.5 if st.selectbox("Overall Height (X5)", ["Single Story (3.5m)", "Double Story (7.0m)"]) == "Single Story (3.5m)" else 7.0
            x6 = int(st.selectbox("Orientation (X6)", ["North (2)", "East (3)", "South (4)", "West (5)"])[-2])
            x7 = float(st.selectbox("Glazing Area (X7)", ["0% (0.0)", "10% (0.1)", "25% (0.25)", "40% (0.4)"]).split('%')[0]) / 100
            x8 = int(st.selectbox("Glazing Distribution (X8)", [
                "Uniform (0)", "North (1)", "East (2)", "South (3)", "West (4)"
            ])[-2])

        submitted = st.form_submit_button("🔍 Predict Energy Efficiency")

    if submitted:
        features = np.array([[x1, x2, x3, x4, x5, x6, x7, x8]])
        scaled = scaler.transform(features)
        pred = model.predict(scaled)[0]

        result_map = {
            0: ("✅ Low Heating Load (High Efficiency)", "green"),
            1: ("⚠️ Medium Heating Load (Moderate Efficiency)", "orange"),
            2: ("❌ High Heating Load (Low Efficiency)", "red")
        }

        text, color = result_map[pred]
        st.markdown(f"<div style='color:{color}; font-weight:bold; font-size:22px'>{text}</div>", unsafe_allow_html=True)

# ========== MODEL DEVELOPMENT ==========
elif st.session_state.page == "Dev":
    st.header("🛠️ Model Development Details")

    st.markdown("### 🔧 Workflow Overview")
    st.markdown("""
    ```
    📥 Input Features (X1–X8)
        ↓
    🧹 StandardScaler (Preprocessing)
        ↓
    🌲 Random Forest Classifier (Tuned with RandomizedSearchCV)
        ↓
    🎯 Output: Energy Efficiency Classification
    ```
    """)

    st.markdown("### ⚙️ Hyperparameter Tuning (RandomizedSearchCV)")
    import pandas as pd
    tuning_data = {
        "Parameter": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
        "Best Value": [500, 20, 2, 1, "log2"]
    }
    st.table(pd.DataFrame(tuning_data))

    st.markdown("### 📊 Model Evaluation")

    st.image("images/tuned_confusion_matrix.png", caption="Figure 1: Confusion Matrix – Tuned Random Forest")

    st.markdown("""
    The tuned model significantly reduced misclassification of moderate-efficiency buildings (class 1),
    compared to the baseline version.
    """)
    st.image("images/accuracy_comparison.png", caption="Figure 2: Accuracy Comparison – Default vs Tuned Random Forest")
    st.markdown("""
    After hyperparameter tuning, model accuracy improved from **94.14% to 94.40%**
    """)

    st.image("images/feature_importance.png", caption="Figure 3: Feature Importances – Tuned Random Forest")
    st.markdown("""
    The most influential features include:
    - **X1: Surface_Area**
    - **X2: Relative_Compactness**
    - **X7: Glazing_Area**

    These features contribute most significantly to classifying building energy efficiency.
    """)

