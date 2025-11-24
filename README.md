## 💳 Credit Card Fraud Detection

A complete end–to–end Machine Learning project to detect fraudulent credit card transactions using highly imbalanced datasets.This project covers data preprocessing, class imbalance handling, model training, evaluation, and comparison using advanced ML techniques.

------------------------------------------------------------------------

## 📘 Project Overview
Credit card fraud is a major challenge for financial institutions. Fraudulent transactions are rare (often less than 0.2%), making this a classic imbalanced classification problem.
This project builds an ML pipeline to detect fraud in real-world transaction data using:
-  Data cleaning
-  Exploratory Data Analysis (EDA)
-  Handling imbalance (SMOTE / undersampling)
-  ML model training
-  Model evaluation
-  Feature scaling
-  Performance comparison

------------------------------------------------------------------------

## 📂 Dataset
The dataset used is a standard Credit Card Fraud Detection Dataset, typically containing:
```
  Feature          Description                                             
  -------------   ------------------------------------------------------- 
  `Time`          Seconds elapsed between transaction & first transaction 
  `V1` to `V28`   PCA-transformed features (sensitive features masked)    
  `Amount`        Transaction amount                                      
  `Class`         **1 = Fraud**, **0 = Legitimate**                       

```
⚠️ Highly imbalanced dataset: Fraud cases are extremely rare.

------------------------------------------------------------------------

## 🧹 Data Preprocessing

✔ Steps included in your notebook:
-  Removing duplicates
-  Checking & handling missing values
-  Feature scaling using StandardScaler
-  Splitting into Train/Test (typically 80/20)
-  Handling class imbalance using:
    -  SMOTE (Synthetic Minority Oversampling)
    -  Random Undersampling
    -  Or both combined (SMOTE + Tomek Links / SMOTEENN)

------------------------------------------------------------------------

## 🧰 Libraries Used
   -   numpy
   -   pandas
   -   matplotlib
   -   seaborn
   -   scikit-learn
------------------------------------------------------------------------

## 🧠 Machine Learning Models Used

-  Logistic Regression
-  Decision Tree Classifier
-  Random Forest Classifier
-  K-Nearest Neighbors (KNN)
-  Support Vector Machine (SVM)
-  XGBoost / Gradient Boosting (if present in code)
-  Naïve Bayes (if included)

Each model is evaluated on both original and balanced datasets.

------------------------------------------------------------------------

## Exploratory Data Analysis
Analyses include:
-  Fraud vs non-fraud transaction ratio
-  Distribution of transaction amounts
-  Correlation analysis
-  Heatmaps for understanding feature impact
-  Visualization of fraud patterns

------------------------------------------------------------------------
## 📈 Model Evaluation Metrics
Models are assessed using:
-  Accuracy
-  Precision (very important for fraud detection)
-  Recall (catching as many frauds as possible)
-  F1 Score
-  Confusion Matrix
-  ROC-AUC Score

Visualizations include:
-  Confusion matrix heatmaps
-  ROC curves
-  Precision-Recall curves

------------------------------------------------------------------------


## 🏆 Best Performing Model
Random Forest achieved ~99% accuracy with high fraud detection precision. 

------------------------------------------------------------------------

## ▶️ How to Run

1️⃣ Install dependencies
```
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost

```
2️⃣ Open Jupyter Notebook
```
jupyter notebook "Credit Card Fraud Detection.ipynb"

```
3️⃣ Run all cells to train and evaluate the model.

------------------------------------------------------------------------
## 🔍 Example ML Pipeline
```
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train model
model = RandomForestClassifier()
model.fit(X_resampled, y_resampled)

# Predict
y_pred = model.predict(X_test)

```
------------------------------------------------------------------------
## 🌟 Future Improvements
- Use deep learning models (LSTM / Autoencoders for anomaly detection)
- Add Explainability using SHAP or LIME
- Deploy model using:
  -  Streamlit
  -  Flask / FastAPI
  -  REST API endpoints

-  Convert pipeline into a Pickle (.pkl) model
-  Implement real-time fraud detection system
  
-----------------------------------------------------------------------
## 👩‍💻 Author

**Monika A.D**
• AI & DS Student
(2025)

------------------------------------------------------------------------

Enjoy learning  🚀
