from flask import Flask, request, jsonify
import pandas as pd
from openai import AzureOpenAI

# Initialize Flask app
app = Flask(__name__)

def load_deductions_data():
    return pd.read_csv("deductions.csv")

def save_deductions_data(df):
    df.to_csv("deductions.csv", index=False)

@app.route("/post_deduction", methods=["POST"])
def add_deduction():
    df = load_deductions_data()
    new_deduction = request.get_json()

    required_fields = ["amount", "reason_code", "Deduction_id", "closing_date", 
                       "cust_name", "cust_number", "Time_to_resolve", "compiled"]
    
    if not all(field in new_deduction for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400
    
    df.loc[len(df)] = new_deduction
    save_deductions_data(df)
    
    return jsonify({"message": "Deduction added successfully"}), 201

@app.route("/update_deduction/<int:deduction_id>", methods=["PUT"])
def update_deduction(deduction_id):
    df = load_deductions_data()
    update_data = request.get_json()

    if deduction_id not in df["Deduction_id"].values:
        return jsonify({"error": "Deduction ID not found"}), 404

    index = df[df["Deduction_id"] == deduction_id].index[0]
    for key, value in update_data.items():
        if key in df.columns:
            df.at[index, key] = value

    save_deductions_data(df)
    return jsonify({"message": "Deduction updated successfully"}), 200

@app.route("/get_unflagged_deductions_by_cust_name/<cust_name>", methods=["GET"])
def get_unflagged_deductions_by_cust_name(cust_name):
    df = load_deductions_data()
    filtered_df = df[(df["cust_name"] == cust_name) & (df["compiled"] == "N")]

    if filtered_df.empty:
        return jsonify({"error": "No unflagged deductions found for this customer"}), 404

    return jsonify(filtered_df.to_dict(orient="records")), 200

@app.route("/get_all_unflagged_deductions", methods=["GET"])
def get_all_unflagged_deductions():
    df = load_deductions_data()
    filtered_df = df[df["compiled"] == "N"]

    if filtered_df.empty:
        return jsonify({"error": "No unflagged deductions available"}), 404

    return jsonify(filtered_df.to_dict(orient="records")), 200

@app.route("/get_deductions_by_amount_greater_than/<int:amount>", methods=["GET"])
def get_deductions_by_amount_greater_than(amount):
    df = load_deductions_data()
    filtered_df = df[df["amount"] > amount]

    if filtered_df.empty:
        return jsonify({"error": "No deductions found with amount greater than the specified value"}), 404

    return jsonify(filtered_df.to_dict(orient="records")), 200

@app.route("/get_deduction_by_id/<int:deduction_id>", methods=["GET"])
def get_deduction_by_id(deduction_id):
    df = load_deductions_data()
    deduction = df[df["Deduction_id"] == deduction_id]

    if deduction.empty:
        return jsonify({"error": "Deduction ID not found"}), 404

    return jsonify(deduction.to_dict(orient="records")[0]), 200

@app.route("/get_deduction_summary", methods=["POST"])
def get_deduction_summary():
    deduction_data = request.get_json()
    
    if not deduction_data:
        return jsonify({"error": "No deduction data provided"}), 400

    system_prompt = (
        "You are an AI assistant that provides summaries of financial deductions. "
        "Analyze the given deduction details and generate a brief summary including whether to give the deductions or not.."
    )
    is_valid_deductions = ""
    user_prompt = f"Deduction Details: {deduction_data}. The Previous AI reponse to provide the deduction to user based on historical Data is {is_valid_deductions}. Provide a reason as well why AI might have selected this reponse. Generate a concise summary."

    summary = azure_api_response(system_prompt, user_prompt)
    
    return jsonify({"deduction_summary": summary}), 200

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True, port=5003)
