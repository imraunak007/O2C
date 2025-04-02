import requests
import json

BASE_URL = "http://127.0.0.1:5002"

# Sample deduction data
deduction_data = {
    "amount": 500,
    "reason_code": "Rebate - Variable Promo",
    "deduction_id": "12345678",
    "closing_date": "03-10-2019",
    "cust_name": "TEST CUSTOMER",
    "cust_number": "0450019999",
    "Time_to_resolve": 200,
    "compiled": "N"
}

# 1️⃣ POST - Add a new deduction
print("\n➡️ Adding Deduction...")
post_response = requests.post(f"{BASE_URL}/post_deduction", json=deduction_data)
print("Response:", post_response.json())

# 2️⃣ PUT - Update the deduction
update_data = {"amount": 600, "compiled": "Y"}
print("\n➡️ Updating Deduction...")
put_response = requests.put(f"{BASE_URL}/update_deduction/12345678", json=update_data)
print("Response:", put_response.json())

# 3️⃣ GET - Get unflagged deductions by customer name
print("\n➡️ Fetching unflagged deductions for TEST CUSTOMER...")
get_customer_response = requests.get(f"{BASE_URL}/get_unflagged_deductions_by_cust_number/450019173")
print("Response:", get_customer_response.json())

# 4️⃣ GET - Get all unflagged deductions
print("\n➡️ Fetching all unflagged deductions...")
get_unflagged_response = requests.get(f"{BASE_URL}/get_all_unflagged_deductions")
print("Response:", get_unflagged_response.json())

# 5️⃣ GET - Get deductions with amount greater than 400
print("\n➡️ Fetching deductions where amount > 400...")
get_amount_response = requests.get(f"{BASE_URL}/get_deductions_by_amount_greater_than/400")
print("Response:", get_amount_response.json())

# 6️⃣ GET - Get deduction by Deduction ID
print("\n➡️ Fetching deduction by Deduction ID (12345678)...")
get_deduction_response = requests.get(f"{BASE_URL}/get_deduction_by_id/12345678")
print("Response:", get_deduction_response.json())

# 7️⃣ POST - Get AI-generated summary for the deduction
print("\n➡️ Getting AI Summary for Deduction ID (12345678)...")
summary_payload = {"cust_number": 450019173, 
                   "reason_code": "Rebate - Variable Promo", 
                   "total_invoice_amount": 117002.03, 
                   "deduction_amount": 1172.03}
summary_response = requests.post(f"{BASE_URL}/get_deduction_summary", json=summary_payload)
print("Response:", summary_response.json())

