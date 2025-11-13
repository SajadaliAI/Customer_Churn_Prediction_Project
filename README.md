# 🔮 Customer Churn Prediction using Machine Learning

This project predicts whether a customer will **churn (leave)** or **stay** with a telecom company based on various factors like contract type, payment method, and monthly charges.

---

## 🎯 Objective
To build a machine learning model that can accurately predict **customer churn** using historical customer data.  

The goal is to help businesses **identify at-risk customers** and take preventive actions.

---

## 🧩 Dataset Overview
Dataset: `Telco_Customer_Churn.csv`

**Columns include:**
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`  
- `tenure`, `PhoneService`, `MultipleLines`  
- `InternetService`, `OnlineSecurity`, `OnlineBackup`  
- `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`  

---

## 🔍 Steps Performed

### 🧹 1. Data Preprocessing
- Handled missing and incorrect values  
- Encoded categorical variables (LabelEncoder / OneHotEncoder)  
- Scaled numerical features (StandardScaler / MinMaxScaler)  

### 📊 2. Exploratory Data Analysis (EDA)
- Visualized churn distribution  
- Analyzed relation between features and churn  
- Found important patterns using correlation heatmaps  

### 🤖 3. Model Training
Implemented and compared:
- Logistic Regression  
  

### 📈 4. Model Evaluation
- Accuracy, Precision, Recall, F1-Score, AUC  
- Confusion Matrix visualization  
- ROC Curve  

### 💾 5. Model Saving
Saved the best model using:
```python
import joblib
joblib.dump(model, 'models/churn_model.pkl')
