1. Introduction
Welcome to the user manual for the Energy Efficiency Classifier application. This guide walks you through setting up the project on your local machine, launching the Streamlit app, and using it to classify the energy efficiency of buildings.

This application is designed to demonstrate how machine learning can be used for fast and reliable building performance prediction based on architectural parameters.

-----

2. Installation and Setup
Follow these steps to set up the application and its dependencies on your computer.

Step 2.1: Get the Project Files
To begin, download the project files from the GitHub repository.

Open your terminal (Command Prompt, PowerShell, or Terminal).

Navigate to the directory where you want to store the project.

Clone the repository using the following command:

git clone <your-repository-url>
Navigate into the cloned project folder:

cd <your-repository-folder>

Step 2.2: Set Up the Virtual Environment
⚠️ Important: This project requires Python 3.11. Check your version:

python --version
If you don’t have Python 3.11, download it from:
👉 https://www.python.org/downloads/release/python-3110/

Create and activate a virtual environment:

macOS/Linux:

python3.11 -m venv .venv
source .venv/bin/activate
Windows:

py -3.11 -m venv .venv
.\.venv\Scripts\activate
You should now see (.venv) in your terminal prompt, indicating that your environment is active.

Step 2.3: Install Required Libraries
Install the required dependencies using:

pip install -r requirements.txt
This installs packages like pandas, scikit-learn, streamlit, joblib, and others necessary for the application.

-----

3. Launching the Application
Once setup is complete:

Ensure you're in the main project directory.

Activate your virtual environment (.venv should be visible in the terminal).

Start the application using:

streamlit run app/app.py
This will launch a local web server, and the app should open automatically in your browser.

-----

4. Navigating the Application
The interface is organized into three main sections accessible from the sidebar.

4.1. 🏠 Home
This section introduces the app’s purpose and its relevance to sustainable development and energy efficiency. It includes brief project goals and background information.

4.2. 🔎 Try the Model
This is where you can interact with the model. Enter building parameters and get a real-time classification of energy efficiency.

4.3. 📈 Model Development
This page outlines the machine learning development process. It includes:

    -Workflow diagram

    -Hyperparameter tuning table

    -Accuracy comparison (baseline vs tuned)

    -Confusion matrix and classification performance

    -Feature importance visualization

-----

5. How to Get a Prediction
To classify a building’s energy efficiency:

5.1 Go to the "🔎 Try the Model" page.

5.2 Enter values for the following building features:

    Relative Compactness

    Surface Area

    Wall Area

    Roof Area

    Overall Height

    Orientation

    Glazing Area

    Glazing Area Distribution

5.3 Click the "Predict Efficiency" button.

5.4 The app will display:

    Predicted Label: Efficient / Moderate / Inefficient

    Advisory Message: General feedback based on the result

-----

6. Troubleshooting
❗ "Model or scaler not found"
If the app fails to start due to missing .pkl files:

Solution: Ensure the trained model and scaler are available in your app/ or models/ directory:

best_random_forest_model.pkl
scaler.pkl

If they are missing:
    1. Re-run the training scripts or notebooks used in development.
    2. Save the model and scaler using joblib.dump() as follows:

    joblib.dump(model, "best_random_forest_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    Then restart the app.