from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and training features
model = joblib.load("model/Churn_model.pkl")
feature_cols = joblib.load("model/feature_cols.pkl")  # training ke columns

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form.to_dict()

    # Convert numeric fields
    numeric_fields = ['gender','SeniorCitizen','Partner','Dependents',
                      'tenure','PhoneService','MonthlyCharges','TotalCharges','PaperlessBilling']
    for key in numeric_fields:
        if key in form_data:
            form_data[key] = float(form_data[key])

    # Convert to DataFrame
    df = pd.DataFrame([form_data])

    # One-hot encode categorical fields
    df_encoded = pd.get_dummies(df)

    # Add missing columns with 0
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns as per training
    df_encoded = df_encoded[feature_cols]

    # Make prediction
    pred = model.predict(df_encoded)[0]          # 0 or 1
    prob = model.predict_proba(df_encoded)[0][1] # Probability of Churn

    result = "Churn" if pred == 1 else "No Churn"
    probability = round(prob*100, 2)

    return render_template('index.html', result=result, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
