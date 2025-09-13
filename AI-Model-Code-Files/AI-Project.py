#Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
# Import tkinter for Yes/No button pop-up Prof this i used just to make you laugh 
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

#Load Dataset and Create a DataFrame
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Dataset shape:", df.shape)
print("Target distribution:\n", df['target'].value_counts())
print("Target labels:", data.target_names)

#Feature/Target Split
X = df.drop('target', axis=1)
y = df['target']

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Standardize Features (important for Logistic Regression and SVMs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# MODEL 1: Logistic Regression (Baseline Model)

lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict probabilities and labels
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]


# MODEL 2: Random Forest (Tuned Model with Hyperparameter Tuning)


# Hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

# GridSearchCV to find best hyperparameters
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model after tuning
rf_model = grid_search.best_estimator_
rf_model.fit(X_train_scaled, y_train)

# Predictions
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]


# EVALUATION METRICS


# Function to print and return all metrics
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n=== {name} Metrics ===")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("AUC:", auc)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=data.target_names))
    return [acc, prec, rec, f1, auc]

lr_scores = evaluate_model("Logistic Regression", y_test, lr_pred, lr_proba)
rf_scores = evaluate_model("Random Forest (Tuned)", y_test, rf_pred, rf_proba)


# CONFUSION MATRICES

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression Confusion Matrix
cm_lr = confusion_matrix(y_test, lr_pred)
disp_lr = ConfusionMatrixDisplay(cm_lr, display_labels=data.target_names)
disp_lr.plot(ax=axes[0])
axes[0].set_title("Logistic Regression")

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=data.target_names)
disp_rf.plot(ax=axes[1])
axes[1].set_title("Random Forest (Tuned)")

plt.tight_layout()
plt.show()


# ROC Curve Comparison

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression", color='orange')
plt.plot(fpr_rf, tpr_rf, label="Random Forest (Tuned)", color='blue')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_comparison.png")
plt.show()


#BAR CHART COMPARISON

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='orange')
plt.bar(x + width/2, rf_scores, width, label='Random Forest (Tuned)', color='blue')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, metrics)
plt.ylim(0.9, 1.01)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("metrics_comparison.png")
plt.show()


# FEATURE IMPORTANCE PLOT for Random Forest 

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]
feature_names = X.columns[indices]

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices], color='green')
plt.yticks(range(len(indices)), feature_names)
plt.xlabel("Feature Importance Score")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
plt.show()

# Function to show image in new window 
def show_image():
    window = tk.Toplevel()
    window.title("Breast Cancer Visual Examples")

    # Load and show image 
    image = Image.open("gr1_lrg.jpg")
    image = image.resize((600, 400), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(image)

    label = tk.Label(window, image=img)
    label.image = img  
    label.pack()

# Create main window for the prompt
root = tk.Tk()
root.withdraw()  

# Ask the Professor(if you click yes Prof you get to see free 8oo8s)
response = messagebox.askquestion("Visual Examples", "Would you like to see example breast cancer images?")

if response == 'yes':
    root.deiconify() 
    show_image()
    root.mainloop()
else:
    print("Okay, closing application.")