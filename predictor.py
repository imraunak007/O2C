import pandas as pd
import joblib
from io import StringIO
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Azure Storage Details
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=o2chub1467504494;AccountKey=ZAdBOJyGCr6mhQpD9R+IlVRIhrMgwNIEfs7gK2y9+xb/bwIQQumK387JeSFfppn/dhP5vZM+odXv+AStdovU6Q==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "o2c-data"

# Initialize Azure Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

def load_data_from_azure():
    """Loads the CSV file directly from Azure Blob Storage into a Pandas DataFrame."""
    blob_name = "collections_formatted_data.csv"
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        csv_data = blob_client.download_blob().content_as_text()
        df = pd.read_csv(StringIO(csv_data))
        print("CSV file loaded successfully from Azure.")
        return df
    except Exception as e:
        print(f"Failed to load CSV from Azure: {e}")
        return None

def upload_to_azure(local_file_path, blob_name):
    """Uploads a local file to Azure Blob Storage."""
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"Uploaded {blob_name} to Azure Storage.")
    except Exception as e:
        print(f"Failed to upload {blob_name}: {e}")


def download_model_from_azure(blob_name, local_file_path):
    """Downloads model or encoder file from Azure Storage."""
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    try:
        with open(local_file_path, "wb") as file:
            file.write(blob_client.download_blob().readall())
        print(f"Downloaded {blob_name} from Azure Storage.")
        return True
    except Exception:
        print(f"{blob_name} not found in Azure. It will be created.")
        return False

class PaymentDatePredictor:
    def __init__(self):
        self.model = None
        self.le = None
        self.data = load_data_from_azure()
        self.load_or_train_model()

    def load_or_train_model(self):
        """Loads the LabelEncoder and Model from Azure Storage or trains a new one."""
        encoder_downloaded = download_model_from_azure("label_encoder.pkl", "label_encoder.pkl")
        model_downloaded = download_model_from_azure("model.pkl", "model.pkl")

        if encoder_downloaded:
            self.le = joblib.load("label_encoder.pkl")
            print("LabelEncoder loaded from Azure.")
        else:
            self.le = LabelEncoder()
            print("No LabelEncoder found. A new one will be trained.")

        if model_downloaded:
            self.model = joblib.load("model.pkl")
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

        df['cust_number_encoded'] = self.le.fit_transform(df['cust_number'])
        joblib.dump(self.le, "label_encoder.pkl")
        upload_to_azure("label_encoder.pkl", "label_encoder.pkl")

        known_data = df[~missing_mask]
        X = known_data[['cust_number_encoded']]
        y = known_data['date_diff(cd-dd)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        joblib.dump(self.model, "model.pkl")
        upload_to_azure("model.pkl", "model.pkl")
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

        cust_encoded = self.le.transform([cust_number])[0]
        predicted_days = int(round(self.model.predict([[cust_encoded]])[0]))
        predicted_payment_date = due_date + timedelta(days=predicted_days)

        return {
            "cust_number": cust_number,
            "due_date": due_date_str,
            "predicted_days": predicted_days,
            "predicted_payment_date": predicted_payment_date.strftime("%d-%m-%Y")
        }

# Example Usage
# if __name__ == "__main__":
#     predictor = PaymentDatePredictor()
#     result = predictor.predict("CUST123", "10-04-2025")
#     print(result)
