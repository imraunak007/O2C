from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from predictor import PaymentDatePredictor
import pandas as pd
from flask import Flask, request, jsonify
from openai import AzureOpenAI
from datetime import datetime
from azure.storage.blob import BlobServiceClient

endpoint = "https://o2chub8437726184.openai.azure.com/"
deployment = "gpt-4o-mini"
api_key = "9FSXgEn7eWvjldGrU54sZ9wsSAtSLGVs0f5pGRoymcjcDwTUQJ3hJQQJ99BCACHYHv6XJ3w3AAAAACOGaDLr"
api_version = "2024-12-01-preview"
STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=o2chub1467504494;AccountKey=ZAdBOJyGCr6mhQpD9R+IlVRIhrMgwNIEfs7gK2y9+xb/bwIQQumK387JeSFfppn/dhP5vZM+odXv+AStdovU6Q==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "o2c-data"
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)
predictor = PaymentDatePredictor()
port = 8001
app = Flask(__name__)

def read_csv_from_blob(blob):
    blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob)
    csv_data = blob_client.download_blob().content_as_text()
    df = pd.read_csv(pd.io.common.StringIO(csv_data))
    return df

def write_csv_to_blob(df, blob):
    blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob)
    csv_data = df.to_csv(index=False)
    blob_client.upload_blob(csv_data, overwrite=True)

data = read_csv_from_blob('dataset.csv')

def load_deductions_data():
    return read_csv_from_blob('deductions.csv')

def is_valid_deduction_amount(deduction_amount, total_invoice_amount):
    return deduction_amount < (0.06 * total_invoice_amount)

def construct_deduction_prompt(historical_deductions, new_reason):
    historical_str = "\n".join([f"{reason}: {freq}" for reason, freq in historical_deductions.items()])
    prompt = f"""
    Given the following list of historical deduction reasons along with their frequency and a new deduction reason, determine if the new reason is a valid deduction based on past records. Respond with only 'Yes' or 'No'.

    Historical Deductions:
    {historical_str}

    New Deduction Reason:
    {new_reason}
    """
    return prompt

def azure_api_response(system_prompt, user_prompt):
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = client.chat.completions.create(
            messages=messages,
            max_tokens=500,
            temperature=0.1,
            top_p=1.0,
            model=deployment
        )
        
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content
        else:
            return "Error: No response from OpenAI."
    
    except Exception as e:
        return f"Error: {str(e)}"

def load_deductions_data():
    return read_csv_from_blob("deductions.csv")

def save_deductions_data(df):
    write_csv_to_blob(df, "deductions.csv")

@app.route("/")
def home():
    return "Welcome to the Invoice Management System!"

@app.route("/post_deduction", methods=["POST"])
def add_deduction():
    df = load_deductions_data()
    new_deduction = request.get_json()
    required_fields = ["amount", "reason_code", "deduction_id", "closing_date", "cust_name", "cust_number"]
    
    if not all(field in new_deduction for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400
    
    if df["deduction_id"].isin([new_deduction["deduction_id"]]).any():
        return jsonify({"error": f"Deduction ID {new_deduction['deduction_id']} already exists. Please provide a unique Deduction ID."}), 400
    df.loc[len(df)] = new_deduction
    save_deductions_data(df)
    return jsonify({"message": "Deduction added successfully"}), 201

@app.route("/update_deduction/<int:deduction_id>", methods=["PUT"])
def update_deduction(deduction_id):
    df = load_deductions_data()
    update_data = request.get_json()

    if deduction_id not in df["deduction_id"].values:
        return jsonify({"error": "Deduction ID not found"}), 404

    index = df[df["deduction_id"] == deduction_id].index[0]
    for key, value in update_data.items():
        if key in df.columns:
            df.at[index, key] = value

    save_deductions_data(df)
    return jsonify({"message": "Deduction updated successfully"}), 200

@app.route("/get_unflagged_deductions_by_cust_name/<cust_name>", methods=["GET"])
def get_unflagged_deductions_by_cust_number(cust_name):
    df = load_deductions_data()
    filtered_df = df[(df["cust_name"] == cust_name) & (df["compiled"].isna() | df["compiled"] != "Y")]
    if filtered_df.empty:
        return jsonify({"error": "No unflagged deductions found for this customer"}), 404
    filtered_df = filtered_df[['deduction_id', 'cust_name', 'amount']]
    return str(filtered_df.replace({pd.NA: None, np.nan: None}).to_dict(orient='records'))

@app.route("/get_all_unflagged_deductions", methods=["GET"])
def get_all_unflagged_deductions():
    df = load_deductions_data()
    filtered_df = df[df["compiled"] == "N"]

    if filtered_df.empty:
        return jsonify({"error": "No unflagged deductions available"}), 404

    return jsonify(filtered_df.replace({pd.NA: None, np.nan: None}).to_dict(orient='records')), 200

@app.route("/get_n_unflagged_deductions", methods=["GET"])
def get_n_unflagged_deductions():
    try:
        n = request.args.get("n", default=10, type=int)
        if n <= 0 or n > 100:
            return jsonify({"error": "Invalid 'n'. Please provide a value between 1 and 100."}), 400

        df = load_deductions_data()
        unflagged_df = df[(df["compiled"].isna()) | (df["compiled"] != "Y")]

        if unflagged_df.empty:
            return jsonify({"error": "No unflagged deductions found."}), 404

        # Optional: sort by latest closing_date if available
        if "closing_date" in unflagged_df.columns:
            unflagged_df = unflagged_df.sort_values(by="closing_date", ascending=False)

        top_n = unflagged_df.head(n)
        top_n = top_n[['deduction_id', 'cust_name', 'amount']]
        return str(top_n.replace({pd.NA: None, np.nan: None}).to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_deductions_by_amount_greater_than/<int:amount>", methods=["GET"])
def get_deductions_by_amount_greater_than(amount):
    df = load_deductions_data()
    filtered_df = df[df["amount"] > amount]

    if filtered_df.empty:
        return jsonify({"error": "No deductions found with amount greater than the specified value"}), 404
    filtered_df = filtered_df[['deduction_id', 'cust_name', 'amount']]
    return str(filtered_df.replace({pd.NA: None, np.nan: None}).to_dict(orient='records'))

@app.route("/get_deduction_by_id/<int:deduction_id>", methods=["GET"])
def get_deduction_by_id(deduction_id):
    df = load_deductions_data()
    deduction = df[df["deduction_id"] == deduction_id]

    if deduction.empty:
        return jsonify({"error": "Deduction ID not found"}), 404

    return jsonify(deduction.replace({pd.NA: None, np.nan: None}).to_dict(orient='records')), 200

@app.route("/get_deduction_summary", methods=["POST"])
def get_deduction_summary():
    deduction_data = request.get_json()
    
    if not deduction_data:
        return jsonify({"error": "No deduction data provided"}), 400

    system_prompt = (
        "You are an AI assistant that provides summaries of financial deductions. "
        "Analyze the given deduction details and generate a brief summary including whether to give the deductions or not."
    )
    with app.test_client() as client:
        response = client.post('/is-valid-deduction', json={"cust_number": deduction_data.get("cust_number"), "reason_code": deduction_data.get("reason_code"), "total_invoice_amount": deduction_data.get("total_invoice_amount"), "deduction_amount": deduction_data.get("deduction_amount")})
        is_valid_deductions = response.get_json()
    user_prompt = f"Deduction Details: {deduction_data}. The Previous AI reponse to provide the deduction to user based on historical Data is {is_valid_deductions}. Provide a reason as well why AI might have selected this reponse. Generate a concise summary."

    summary = azure_api_response(system_prompt, user_prompt)
    
    return jsonify({"deduction_summary": summary}), 200

@app.route("/get_deduction_summary_id/<int:deduction_id>", methods=["GET"])
def get_deduction_summary_by_id(deduction_id):
    try:
        # Load deductions data
        df = load_deductions_data()
        row = df[df["deduction_id"] == deduction_id]

        if row.empty:
            return jsonify({"error": "Deduction ID not found."}), 404

        # Extract necessary fields
        deduction_data = row.iloc[0].to_dict()

        required_fields = ["cust_number", "reason_code", "amount", "total_invoice_amount"]
        if not all(field in deduction_data for field in required_fields):
            return jsonify({"error": "Required deduction fields are missing in the data."}), 400

        # Prepare payload for validity check
        validity_payload = {
            "cust_number": deduction_data["cust_number"],
            "reason_code": deduction_data["reason_code"],
            "deduction_amount": deduction_data["amount"],
            "total_invoice_amount": deduction_data["total_invoice_amount"]
        }

        # Use test client to call the validity check endpoint
        with app.test_client() as client:
            response = client.post('/is-valid-deduction', json=validity_payload)
            validity_result = response.get_json()

        # Construct prompt for GPT
        system_prompt = (
            "You are an AI assistant that provides summaries of financial deductions. "
            "Analyze the given deduction details and generate a brief summary including whether to give the deductions or not."
        )
        user_prompt = (
            f"Deduction Details: {deduction_data}. "
            f"The Previous AI response based on historical Data is {validity_result}. "
            f"Provide a reason as well why AI might have selected this response. Generate a concise summary."
        )

        summary = azure_api_response(system_prompt, user_prompt)
        return jsonify(summary), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/is-valid-deduction', methods=['POST'])
def is_valid_deduction():
    data = request.get_json()
    required_fields = ['cust_number', 'reason_code', 'total_invoice_amount', 'deduction_amount']

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields."}), 400
    
    cust_number = data['cust_number']
    reason_code = data['reason_code']
    total_invoice_amount = float(data['total_invoice_amount'])
    deduction_amount = float(data['deduction_amount'])

    if not is_valid_deduction_amount(deduction_amount, total_invoice_amount):
        return jsonify({"result": False}), 200
    
    df = load_deductions_data()
    cust_data = df[df['cust_number'] == cust_number]
    
    if cust_data.empty:
        return jsonify({"result": True}), 200
    
    historical_deductions = cust_data.groupby('reason_code').size().to_dict()
    user_prompt = construct_deduction_prompt(historical_deductions, reason_code)
    system_prompt = ("You are a helpful assistant.")
    response = azure_api_response(system_prompt, user_prompt)
    is_valid = "yes" in str(response).strip().lower()
    return jsonify({"result": is_valid}), 200

@app.route('/get_invoice_by_cust_name', methods=['GET'])
def get_invoice_by_customer_name():
    name_customer = request.args.get('name_customer')
    start_date = request.args.get('start_date')  # YYYYMMDD
    end_date = request.args.get('end_date')      # YYYYMMDD
    if not name_customer or not start_date or not end_date:
        return jsonify({"error": "Please provide name_customer, start_date, and end_date."}), 400
    
    if name_customer not in data['name_customer'].values:
        return jsonify({"error": "Customer not available."}), 404
    
    # Convert to integer for filtering
    start_date, end_date = int(start_date), int(end_date)
    
    filtered = data[(data['name_customer'] == name_customer) & 
                    (data['document_create_date'].astype(int) >= start_date) &
                    (data['document_create_date'].astype(int) <= end_date)]
    if filtered.empty:
        return jsonify({"error": "No data for this customer in the specified time frame. Choose another time frame."}), 404
    filtered = filtered[['invoice_id', 'document_create_date', 'total_open_amount']]
    filtered['invoice_id'] = filtered['invoice_id'].astype(int).astype(str)
    return str(filtered.replace({pd.NA: None, np.nan: None}).to_dict(orient='records'))

@app.route('/get_latest_invoices', methods=['GET'])
def get_latest_invoices():
    try:
        n = request.args.get('n', type=int, default=10)  # Default to 10 if not provided

        if n <= 0 or n > 50:
            return jsonify({"error": "Invalid value for 'n'. It must be between 1 and 50."}), 400

        latest_invoices = data.sort_values(by="document_create_date", ascending=False).head(n)

        if latest_invoices.empty:
            return jsonify({"error": "No invoice data available."}), 404
        latest_invoices = latest_invoices[['invoice_id', 'document_create_date', 'total_open_amount']]
        latest_invoices['invoice_id'] = latest_invoices['invoice_id'].astype(int).astype(str)
        # Convert NaN values to None for valid JSON
        return str(latest_invoices.replace({pd.NA: None, np.nan: None}).to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/post_invoice', methods=['POST'])
def add_invoice():
    try:
        new_invoice = request.get_json()
        if not new_invoice:
            return jsonify({"error": "No data provided."}), 400
        
        required_fields = ["business_code", "cust_number", "name_customer", "buisness_year", "doc_id", 
                           "posting_date", "document_create_date", "due_in_date", "invoice_currency", "document_type", 
                           "posting_id", "total_open_amount", "baseline_create_date", "cust_payment_terms", "invoice_id", "isOpen"]
        
        for field in required_fields:
            if field not in new_invoice:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Check if invoice_id already exists
        if new_invoice["invoice_id"] in data["invoice_id"].values:
            return jsonify({"error": "Invoice ID already exists. Duplicate entries are not allowed."}), 409
        
        # Append new data
        data.loc[len(data)] = new_invoice
        write_csv_to_blob(data, 'dataset.csv')
        
        return jsonify({"message": "Invoice added successfully.", "invoice": new_invoice}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update_invoice', methods=['PUT'])
def update_invoice():
    try:
        update_data = request.get_json()
        if not update_data:
            return jsonify({"error": "No data provided."}), 400
        
        invoice_id = update_data.get("invoice_id")
        if not invoice_id:
            return jsonify({"error": "Invoice id is required."}), 400
        
        if invoice_id not in data['invoice_id'].values:
            return jsonify({"error": "Invoice id not found."}), 404
        
        index = data[data['invoice_id'] == invoice_id].index[0]
        for key, value in update_data.items():
            if key in data.columns:
                data.at[index, key] = value
        write_csv_to_blob(data, 'dataset.csv')
        return jsonify({
            "message": "Invoice details updated successfully.",
            "updated_data": data.loc[index].where(pd.notna(data.loc[index]), None).to_dict()
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/invoice_summary', methods=['POST'])
def get_invoice_summary():
    try:
        invoice_data = request.get_json()
        if not invoice_data:
            return jsonify({"error": "No invoice data provided."}), 400
        system_prompt = (
            "You are an AI assistant that provides concise summaries of invoice details. "
            "Analyze the given invoice JSON and generate a summary. "
            "If 'isOpen' is 0, mark it as 'PAID'. If 'isOpen' is 1, mark it as 'DUE'. "
            "Ensure the response clearly states whether the invoice is paid or unpaid."
        )
        with app.test_client() as client:
            response = client.post('/predict-payment-date', json={"cust_number": invoice_data.get("cust_number"), "due_date": datetime.strptime(str(int(invoice_data.get("due_date"))), "%Y%m%d").strftime("%d-%m-%Y")})
            predicted_payment_date = response.get_json()
        if int(invoice_data.get("isOpen")) == 1:
            user_prompt = f"Invoice Details: {invoice_data}. Predicted Payment Date: {predicted_payment_date}. Generate a summary of the invoice details with predicted payment date."
        else:
            user_prompt = f"Invoice Details: {invoice_data}. Generate a summary of the invoice details."
        summary = azure_api_response(system_prompt, user_prompt)
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_invoice_summary_by_invoice_id/<invoice_id>', methods=['GET'])
def get_invoice_summary_by_invoice_id(invoice_id):
    try:
        # Check if invoice_id exists
        invoice = data.copy()
        invoice['invoice_id'] = invoice['invoice_id'].fillna(-1).astype(int)
        invoice = invoice[invoice['invoice_id'] == int(invoice_id)]
        if invoice.empty:
            return jsonify({"error": "Invoice ID not found."}), 404

        # Convert row to dict
        invoice_record = invoice.iloc[0].to_dict()

        # Use test client to call /invoice_summary
        with app.test_client() as client:
            response = client.post('/invoice_summary', json=invoice_record)
            summary = response.get_json()

        return jsonify(summary), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict-payment-date', methods=['POST'])
def predict_payment_date():
    data = request.json
    cust_number = data.get("cust_number")
    due_date = data.get("due_date")

    if not cust_number or not due_date:
        return jsonify({"error": "Missing cust_number or due_date"}), 400
    result = predictor.predict(cust_number, due_date)
    return jsonify(result)

@app.route('/mitigation-strategies', methods=['POST'])
def get_mitigation_strategies():
    try:
        data = request.get_json()
        if not data or "invoice_details" not in data or "predicted_payment_date" not in data:
            return jsonify({"error": "Missing required fields: 'invoice_details' and 'predicted_payment_date'"}), 400
        invoice_details = data["invoice_details"]
        predicted_payment_date = data["predicted_payment_date"]

        system_prompt = (
            "You are a financial assistant that provides mitigation strategies for overdue invoices. "
            "Based on the invoice details and predicted payment date, suggest steps to recover payment efficiently."
        )

        user_prompt = (
            f"Invoice Details: {invoice_details}. "
            f"The due date was {invoice_details.get('due_in_date')}, but the predicted payment date is {predicted_payment_date}. "
            f"What mitigation strategies should we take?"
        )

        strategies = azure_api_response(system_prompt, user_prompt)
        return jsonify({"mitigation_strategies": strategies})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_mitigation_strategies_by_invoice_id/<invoice_id>', methods=['GET'])
def get_mitigation_strategies_by_invoice_id(invoice_id):
    try:
        invoice_df = data.dropna(subset=['invoice_id'])
        invoice_df = invoice_df[invoice_df['invoice_id'].astype(int) == int(invoice_id)]
        
        if invoice_df.empty:
            return jsonify({"error": "Invoice ID not found."}), 404

        invoice_details = invoice_df.iloc[0].to_dict()
        # Call internal prediction API
        with app.test_client() as client:
            response = client.post('/predict-payment-date', json={
                "cust_number": invoice_details.get("cust_number"),
                "due_date": datetime.strptime(str(int(invoice_details.get("due_date"))), "%Y%m%d").strftime("%d-%m-%Y")
            })
            prediction = response.get_json()
        
        if not prediction or "predicted_payment_date" not in prediction:
            return jsonify({"error": "Failed to predict payment date."}), 500

        # Generate mitigation strategies via LLM
        system_prompt = (
            "You are a financial assistant that provides mitigation strategies for overdue invoices. "
            "Based on the invoice details and predicted payment date, suggest steps to recover payment efficiently."
        )

        user_prompt = (
            f"Invoice Details: {invoice_details}. "
            f"The due date was {invoice_details.get('due_date')}, but the predicted payment date is {prediction['predicted_payment_date']}. "
            f"What mitigation strategies should we take? And add due date and predicted payment date to the response. Keep it short and concise"
        )

        strategies = azure_api_response(system_prompt, user_prompt)

        return jsonify(strategies)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
