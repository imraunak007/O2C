import os
import pandas as pd
import joblib
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Azure Storage Details
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=o2chub1467504494;AccountKey=ZAdBOJyGCr6mhQpD9R+IlVRIhrMgwNIEfs7gK2y9+xb/bwIQQumK387JeSFfppn/dhP5vZM+odXv+AStdovU6Q==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "o2c-data"

# Local File Paths
LOCAL_MODEL_PATH = "model.pkl"
LOCAL_ENCODER_PATH = "label_encoder.pkl"
CSV_FILE_PATH = "collections_formatted_data.csv"

# Initialize Azure Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)


def blob_exists(blob_name):
    """Checks if a blob exists in Azure Storage."""
    return any(blob.name == blob_name for blob in container_client.list_blobs())


def upload_to_azure(local_file_path, blob_name):
    """Uploads a local file to Azure Blob Storage."""
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"Uploaded {blob_name} to Azure Storage.")
    except Exception as e:
        print(f"Failed to upload {blob_name}: {e}")


def download_from_azure(blob_name, local_file_path):
    """Downloads a file from Azure Blob Storage to local storage if it exists."""
    if not blob_exists(blob_name):
        print(f"{blob_name} not found in Azure. It will be created.")
        return False

    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        with open(local_file_path, "wb") as file:
            file.write(blob_client.download_blob().readall())
        print(f"Downloaded {blob_name} from Azure Storage.")
        return True
    except Exception as e:
        print(f"Failed to download {blob_name}: {e}")
        return False


class PaymentDatePredictor:
    def __init__(self):
        self.model = None
        self.le = None
        self.data = None

        # Load CSV Data
        self.load_data()

        # Load or train model
        self.load_or_train_model()

    def load_data(self):
        """Loads the CSV file for training."""
        if not os.path.exists(CSV_FILE_PATH):
            print(f"CSV file '{CSV_FILE_PATH}' not found!")
            return
        self.data = pd.read_csv(CSV_FILE_PATH)
        print("CSV file loaded successfully.")

    def load_or_train_model(self):
        """Loads the LabelEncoder and Model from Azure Storage or trains a new one."""
        encoder_downloaded = download_from_azure("label_encoder.pkl", LOCAL_ENCODER_PATH)
        model_downloaded = download_from_azure("model.pkl", LOCAL_MODEL_PATH)

        # Load LabelEncoder if available, otherwise create a new one
        if encoder_downloaded and os.path.exists(LOCAL_ENCODER_PATH):
            self.le = joblib.load(LOCAL_ENCODER_PATH)
            print("LabelEncoder loaded from Azure.")
        else:
            self.le = LabelEncoder()
            print("No LabelEncoder found. A new one will be trained.")

        # Load Model if available, otherwise train a new one
        if model_downloaded and os.path.exists(LOCAL_MODEL_PATH):
            self.model = joblib.load(LOCAL_MODEL_PATH)
            print("Model loaded from Azure.")
        else:
            print("No Model found. Training a new one...")
            self.train_model()

    def train_model(self):
        """Trains the model and uploads it to Azure Storage."""
        if self.data is None:
            print("No data available for training.")
            return

        df = self.data[['cust_number', 'date_diff(cd-dd)']].copy()
        missing_mask = df['date_diff(cd-dd)'].isnull()

        # Encode customer numbers
        df['cust_number_encoded'] = self.le.fit_transform(df['cust_number'])

        # Save and upload LabelEncoder
        joblib.dump(self.le, LOCAL_ENCODER_PATH)
        upload_to_azure(LOCAL_ENCODER_PATH, "label_encoder.pkl")

        # Prepare training data
        known_data = df[~missing_mask]
        X = known_data[['cust_number_encoded']]
        y = known_data['date_diff(cd-dd)']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train RandomForest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Save and upload model
        joblib.dump(self.model, LOCAL_MODEL_PATH)
        upload_to_azure(LOCAL_MODEL_PATH, "model.pkl")

        print("Model trained and uploaded to Azure.")

    def predict(self, cust_number, due_date_str):
        """Predicts payment date based on customer number and due date."""
        if self.model is None:
            return {"error": "Model not found. Train the model first."}

        try:
            due_date = datetime.strptime(due_date_str, "%d-%m-%Y")
        except ValueError:
            return {"error": "Invalid date format. Use DD-MM-YYYY."}

        if cust_number not in self.le.classes_:
            return {
                "cust_number": cust_number,
                "due_date": due_date_str,
                "predicted_days": 0,
                "predicted_payment_date": due_date_str
            }

        # Encode customer number
        cust_encoded = self.le.transform([cust_number])[0]

        # Predict delay in days
        predicted_days = int(round(self.model.predict([[cust_encoded]])[0]))
        predicted_payment_date = due_date + timedelta(days=predicted_days)

        return {
            "cust_number": cust_number,
            "due_date": due_date_str,
            "predicted_days": predicted_days,
            "predicted_payment_date": predicted_payment_date.strftime("%d-%m-%Y")
        }
