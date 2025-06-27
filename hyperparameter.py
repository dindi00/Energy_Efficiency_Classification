import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv(r"C:\Users\Dinosour\Desktop\ML_PROJECT_NEW\HyperTuned_Version\energy_efficiency_data.csv")

# Create class labels
heat_bins = [0, 15, 24.5, 43.1]
cool_bins = [0, 18.5, 27.5, 48.03]
temp_labels = [0, 1, 2]
df['HeatClass'] = pd.cut(df['Heating_Load'], bins=heat_bins, labels=temp_labels)
df['CoolClass'] = pd.cut(df['Cooling_Load'], bins=cool_bins, labels=temp_labels)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# Features and target
x = df.drop(columns=['Heating_Load', 'Cooling_Load', 'HeatClass', 'CoolClass'])
yh = df['HeatClass']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Train-test split
X_train, X_test, yh_train, yh_test = train_test_split(X_scaled, yh, test_size=0.25, random_state=42)

# ========== BASELINE MODELS ==========
# Decision Tree (optional baseline)
DT = DecisionTreeClassifier(random_state=42)
DT.fit(X_train, yh_train)
DT_pred = DT.predict(X_test)
print('Decision Tree Accuracy: {:.2%}'.format(accuracy_score(yh_test, DT_pred)))

# Default Random Forest
RF_default = RandomForestClassifier(random_state=42)
RF_default.fit(X_train, yh_train)
RF_default_pred = RF_default.predict(X_test)
print('Random Forest Accuracy (default): {:.2%}'.format(accuracy_score(yh_test, RF_default_pred)))

# ========== TUNED RANDOM FOREST ==========
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=30, cv=5, verbose=1,
    random_state=42, n_jobs=-1, scoring='accuracy'
)
random_search.fit(X_train, yh_train)

best_rf = random_search.best_estimator_
yh_pred_optimized = best_rf.predict(X_test)

# Report
print("\nBest Parameters:", random_search.best_params_)
print("Optimized Accuracy: {:.2%}".format(accuracy_score(yh_test, yh_pred_optimized)))
print("\nClassification Report:\n", classification_report(yh_test, yh_pred_optimized))

# Confusion Matrix
cm = confusion_matrix(yh_test, yh_pred_optimized)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_).plot(cmap='Blues')
plt.title("Confusion Matrix - Tuned Random Forest")
plt.show()

# Feature Importance
importances = best_rf.feature_importances_
features = x.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances - Tuned Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ========== ACCURACY COMPARISON ==========
print("\n Accuracy Comparison:")

train_acc_default = accuracy_score(yh_train, RF_default.predict(X_train))
test_acc_default = accuracy_score(yh_test, RF_default_pred)
cv_acc_default = cross_val_score(RF_default, X_scaled, yh, cv=5).mean()

train_acc_tuned = accuracy_score(yh_train, best_rf.predict(X_train))
test_acc_tuned = accuracy_score(yh_test, yh_pred_optimized)
cv_acc_tuned = cross_val_score(best_rf, X_scaled, yh, cv=5).mean()

print(f"Default RF - Train: {train_acc_default:.2%}, Test: {test_acc_default:.2%}, CV: {cv_acc_default:.2%}")
print(f"Tuned RF   - Train: {train_acc_tuned:.2%}, Test: {test_acc_tuned:.2%}, CV: {cv_acc_tuned:.2%}")

# Plot comparison
labels = ['Train Accuracy', 'Test Accuracy', 'CV Accuracy']
default_scores = [train_acc_default, test_acc_default, cv_acc_default]
tuned_scores = [train_acc_tuned, test_acc_tuned, cv_acc_tuned]

x_pos = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x_pos - width/2, default_scores, width, label='Default RF')
bars2 = ax.bar(x_pos + width/2, tuned_scores, width, label='Tuned RF')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison: Default vs Tuned Random Forest')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylim(0.7, 1.0)
ax.legend()
plt.tight_layout()
plt.show()

dataset_path = r"C:\Users\Dinosour\Desktop\ML_PROJECT_NEW\HyperTuned_Version\energy_efficiency_data.csv"
dataset_dir = os.path.dirname(dataset_path)
model_path = os.path.join(dataset_dir, 'random_forest_model.pkl')
scaler_path = os.path.join(dataset_dir, 'scaler.pkl')

joblib.dump(best_rf, model_path)
joblib.dump(scaler, scaler_path)
print(f"✅ Model saved to: {model_path}")
print(f"✅ Scaler saved to: {scaler_path}")
