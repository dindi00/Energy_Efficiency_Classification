# ⚡ ENER-G: Energy Efficiency Classifier

A Streamlit-powered web application that predicts the energy efficiency of buildings based on architectural parameters using a fine-tuned Random Forest classifier.

---

## 🚀 Overview

This project helps users — including engineers, architects, and sustainability enthusiasts — assess a building's energy efficiency at the design stage. Users input 8 building design parameters, and the model classifies the building into one of three energy efficiency levels (low, medium, high heating load).

---

## 🎯 Project Goals

- Predict building energy efficiency from early-stage design features.
- Promote energy-conscious architectural decisions.
- Provide a real-time, user-friendly web interface.
- Align with Sustainable Development Goals (SDGs).

---

## 🛠️ Features

- 🧪 **Predict Now**: Input building specs and receive instant classification.
- 🛠️ **Model Development**: See model workflow, tuning parameters, and evaluation charts.
- 🧾 **Prediction History**: View your previous predictions.
- 📊 **Main Page**: Learn about the tool and its SDG impact.

---

## 🧠 Machine Learning

- **Algorithm**: Random Forest Classifier (sklearn)
- **Tuning**: RandomizedSearchCV with 5-fold cross-validation
- **Accuracy**: Improved from 94.14% → 94.40% after tuning
- **Input Features**:
  - Relative Compactness (X1)
  - Surface Area (X2)
  - Wall Area (X3)
  - Roof Area (X4)
  - Overall Height (X5)
  - Orientation (X6)
  - Glazing Area (X7)
  - Glazing Distribution (X8)

## 🧾 Required Files

- `best_random_forest_model.pkl` – Trained model file
- `scaler.pkl` – Scaler used for preprocessing input features
- `/images/` – Directory containing evaluation visualizations:
  - `tuned_confusion_matrix.png`
  - `accuracy_comparison.png`
  - `feature_importance.png`

---

## ▶️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/energy-efficiency-classifier.git
   cd energy-efficiency-classifier


2. Install dependencies:

pip install -r requirements.txt

3. Launch the Streamlit app:

streamlit run Streamlit.py
## 📁 Project Directory Structure

```
energy_efficiency_classifier/
├── Streamlit.py
├── best_random_forest_model.pkl
├── scaler.pkl
├── appdata/
│   └── energy_prediction_history.csv
├── images/
│   ├── tuned_confusion_matrix.png
│   ├── accuracy_comparison.png
│   └── feature_importance.png
├── README.md
```

👥 Team ENER-G
Aizuddin
Qastalaani
Jack Leonardo
Ahmad Fauzi

---

📌 SDG Alignment
SDG 7 – Affordable and Clean Energy
SDG 9 – Industry, Innovation and Infrastructure
SDG 11 – Sustainable Cities and Communities
SDG 13 – Climate Action

---

📜 License
This project is for educational and demonstration purposes only.



