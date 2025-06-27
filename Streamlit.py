import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# Page Configuration
st.set_page_config(page_title="Energy Efficiency Classifier", page_icon="⚡", layout="wide")



# Team Name
st.markdown("""
<h1 style='text-align:center; font-size:64px; color:#00FFD1; font-weight:900;
            text-shadow: 0 0 10px #00FFD1, 0 0 20px #00FFD1, 0 0 30px #00FFD1;'>
    ⚡ ENER-G ⚡
</h1>
""", unsafe_allow_html=True)

# Team Members Name
st.markdown("""
<p style='text-align:center; font-size:14px; color:#888; margin-top:-20px;'>
Built by Aizuddin, Qastalaani, Jack Leonardo, and Ahmad Fauzi 
</p>
""", unsafe_allow_html=True)



# Paths & History Setup
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR  / 'TunedModels' / 'best_random_forest_model.pkl'
SCALER_PATH = BASE_DIR / 'TunedModels' / 'scaler.pkl'
HISTORY_PATH = BASE_DIR / 'appdata' / 'energy_prediction_history.csv'
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()


# Session State Initialization
if "page" not in st.session_state:
    st.session_state.page = "Home"


# Save Prediction to History
def save_prediction_to_history(input_data, predicted_label):
    record = {
        "Timestamp": datetime.now().isoformat(timespec='seconds'),
        **{f"X{i+1}": val for i, val in enumerate(input_data)},
        "Prediction": predicted_label
    }
    df_record = pd.DataFrame([record])

    if HISTORY_PATH.exists():
        try:
            df_existing = pd.read_csv(HISTORY_PATH)
            df_all = pd.concat([df_existing, df_record], ignore_index=True)
        except pd.errors.EmptyDataError:
            df_all = df_record
    else:
        df_all = df_record

    df_all.to_csv(HISTORY_PATH, index=False)


# Styling
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


# Title
st.markdown("<h2 style='text-align:center;'>🔍 Energy Efficiency Classifier</h2>", unsafe_allow_html=True)


# Navigation Buttons
nav_cols = st.columns([1, 2, 2, 2, 2, 1])

with nav_cols[1]:
    if st.button("📊 Main Page"):
        st.session_state.page = "Home"
with nav_cols[2]:
    if st.button("🤖 Predict Now"):
        st.session_state.page = "Try"
with nav_cols[3]:
    if st.button("🛠️ Model Development"):
        st.session_state.page = "Dev"
with nav_cols[4]:
    if st.button("🧾 Past Prediction"):
        st.session_state.page = "History"

st.markdown("---")


# Home Page
if st.session_state.page == "Home":
    st.header("🏠 Welcome to the Energy Efficiency Classifier")
    st.markdown("""
    This tool classifies the energy efficiency of buildings using a fine-tuned Random Forest model.

    **Steps:**
    - Head over to **Try Model** to input your building's specifications.
    - View how the classifier predicts heating load efficiency.
    """)

    st.markdown("### 📌 Project Objective")
    st.markdown("""
    - Predict building energy efficiency based on design parameters.
    - Assist engineers in optimizing energy usage early in design.
    - Promote sustainable architecture.
    - Provide a live web interface for evaluation.
    """)

    st.markdown("### 🌍 SDG Alignment")
    st.markdown("""
     - **🏡 SDG 7: Affordable and Clean Energy**  
      By promoting energy-efficient building designs, the project reduces dependency on non-renewable energy.

    - **🏗️ SDG 9: Industry, Innovation and Infrastructure**  
      Encourages the use of innovative technologies (AI/ML) in construction and design sectors.

    - **🌱 SDG 11: Sustainable Cities and Communities**  
      Supports eco-friendly building design, contributing to more sustainable urban environments.

    - **♻️ SDG 13: Climate Action**  
      Reducing heating load through better building designs helps lower overall carbon emissions.
    """)

# Try Model Page
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
            x8 = int(st.selectbox("Glazing Distribution (X8)", ["Uniform (0)", "North (1)", "East (2)", "South (3)", "West (4)"])[-2])

        submitted = st.form_submit_button("🔍 Predict Energy Efficiency")

    if submitted:
        features = np.array([[x1, x2, x3, x4, x5, x6, x7, x8]])
        scaled = scaler.transform(features)

        # Predict label and probabilities
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0]

        # Prediction map
        result_map = {
            0: {
                "label": "✅ Low Heating Load (High Efficiency)",
                "color": "green",
                "advice": "Great job! Your building is highly energy-efficient. Keep this design in mind for sustainability."
            },
            1: {
                "label": "⚠️ Medium Heating Load (Moderate Efficiency)",
                "color": "orange",
                "advice": "This is acceptable, but there's room to improve energy performance. Consider optimizing glazing and orientation."
            },
            2: {
                "label": "❌ High Heating Load (Low Efficiency)",
                "color": "red",
                "advice": "Energy performance is poor. Rethink the building's compactness, materials, or insulation strategies."
            }
        }

        # Extract result
        label = result_map[pred]["label"]
        color = result_map[pred]["color"]
        advice = result_map[pred]["advice"]

        # Display
        st.markdown(f"<div style='color:{color}; font-weight:bold; font-size:22px'>{label}</div>", unsafe_allow_html=True)
        st.warning(f"📌 **Advice**: {advice}")

        # Save to history
        save_prediction_to_history([x1, x2, x3, x4, x5, x6, x7, x8], label)


# Model Development Page
elif st.session_state.page == "Dev":
    st.header("🛠️ Model Development Details")

    st.markdown("### 🔧 Workflow Overview")
    st.markdown("""
    ```
    📥 Input Features (X1–X8)
                    ↓
    🧹 StandardScaler (Preprocessing)
                    ↓
    🌲 Random Forest Classifier (Tuned)
                    ↓
    🎯 Output: Energy Efficiency Classification
    ```
    """)

    st.markdown("### ⚙️ Hyperparameter Tuning")
    tuning_data = {
        "Parameter": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
        "Best Value": [500, 20, 2, 1, "log2"]
    }
    st.table(pd.DataFrame(tuning_data))

    st.markdown("### 📊 Model Evaluation")
    st.image(BASE_DIR / "images" / "tuned_confusion_matrix.png", caption="Figure 1: Confusion Matrix")
    st.image(BASE_DIR / "images" / "accuracy_comparison.png", caption="Figure 2: Accuracy Comparison")
    st.markdown("Accuracy improved from **94.14%** to **94.40%** after tuning.")
    st.image(BASE_DIR / "images" / "feature_importance.png", caption="Figure 3: Feature Importances")


# History Page
elif st.session_state.page == "History":
    st.header("🧾 Prediction History")
    if HISTORY_PATH.exists():
        try:
            history_df = pd.read_csv(HISTORY_PATH)
            if not history_df.empty:
                st.dataframe(history_df)
                if st.button("🗑️ Clear History"):
                    HISTORY_PATH.unlink()
                    st.success("Prediction history cleared.")
            else:
                st.info("No predictions have been made yet.")
        except pd.errors.EmptyDataError:
            st.info("No predictions have been made yet.")
    else:
        st.info("No predictions have been made yet.")
