import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

class PaymentDatePredictor:
    def __init__(self, data_path="collections_formatted_data.csv", model_path="model.pkl"):
        self.data_path = data_path
        self.model_path = model_path
        self.data = pd.read_csv(data_path)
        self.le = LabelEncoder()
        self.model = None
        self.prepare_data()
    
    def prepare_data(self):
        """Prepares data for training by encoding customer numbers and handling missing values."""
        df = self.data[['cust_number', 'date_diff(cd-dd)']].copy()
        missing_mask = df['date_diff(cd-dd)'].isnull()
        df['cust_number_encoded'] = self.le.fit_transform(df['cust_number'])
        
        known_data = df[~missing_mask]
        unknown_data = df[missing_mask]

        X = known_data[['cust_number_encoded']]
        y = known_data['date_diff(cd-dd)']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(self.model, self.model_path)

        # Fill missing values
        if not unknown_data.empty:
            X_missing = unknown_data[['cust_number_encoded']]
            missing_predictions = self.model.predict(X_missing).round().astype(int)
            df.loc[missing_mask, 'date_diff(cd-dd)'] = missing_predictions
    
    def predict(self, cust_number, due_date_str):
        """Predicts payment date based on customer name and due date."""
        try:
            due_date = datetime.strptime(due_date_str, "%d-%m-%Y")
        except ValueError:
            return {"error": "Invalid date format. Use DD-MM-YYYY."}

        if cust_number not in self.le.classes_:
            return {"cust_number": cust_number, "due_date": due_date_str, "predicted_days": 0, "predicted_payment_date": due_date_str}
        
        cust_encoded = self.le.transform([cust_number])[0]
        predicted_days = int(round(self.model.predict([[cust_encoded]])[0]))
        predicted_payment_date = due_date + timedelta(days=predicted_days)

        return {
            "cust_number": cust_number,
            "due_date": due_date_str,
            "predicted_days": predicted_days,
            "predicted_payment_date": predicted_payment_date.strftime("%d-%m-%Y")
        }
