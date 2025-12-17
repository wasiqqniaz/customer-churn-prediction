
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load the trained model and preprocessing objects
with open("churn_model.pkl", "rb") as f:
    churn_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("oh_encoder.pkl", "rb") as f:
    oh_encoder = pickle.load(f)

with open("numerical_means.pkl", "rb") as f:
    numerical_means = pickle.load(f)

with open("numerical_medians.pkl", "rb") as f:
    numerical_medians = pickle.load(f)

with open("categorical_modes.pkl", "rb") as f:
    categorical_modes = pickle.load(f)

print("Model, encoder, scaler, and statistics loaded successfully!\n")

# Define column lists
numerical_columns = ['total_orders', 'total_spent', 'avg_order_value',
                     'avg_product_price', 'avg_review_score', 'avg_items_per_order']

categorical_columns = ['preferred_payment_type']

features_columns = ['customer_unique_id', 'total_orders', 'total_spent', 'avg_order_value',
                    'preferred_payment_type', 'avg_product_price', 'avg_review_score',
                    'days_since_last_purchase', 'avg_items_per_order']

dropped_columns = ['customer_unique_id', 'days_since_last_purchase']

# Example new input data
# Format: customer_unique_id,total_orders,total_spent,avg_order_value,preferred_payment_type,avg_product_price,avg_review_score,days_since_last_purchase,avg_items_per_order
# input_data = "CUST001,10,500,50,credit_card,45,4.2,30,2"
# input_data = 'CUST1002,40,4000,100,credit_card,95,5.0,2,5'
input_data = 'C123,5,1200,240,credit_card,60,4.5,30,2'

# Split the input string by commas
values = input_data.split(',')

# Convert to proper types
def enclose_value(value):
    if value == "" or value is None:
        return np.nan
    try:
        if "." in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        return value

processed_values = [enclose_value(v) for v in values]

# Validate input length
if len(processed_values) != len(features_columns):
    raise ValueError("Input data does not match the expected number of features.")

# Convert the processed values into a DataFrame
input_data = pd.DataFrame([processed_values], columns=features_columns)

# Remove dropped columns
if dropped_columns:
    input_df = input_data.drop(columns=dropped_columns, errors='ignore')

# Handle missing values
input_df[numerical_columns] = input_df[numerical_columns].fillna(numerical_means)
input_df[categorical_columns] = input_df[categorical_columns].fillna(categorical_modes)

# Scale numerical columns
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# Encode preferred_payment_type (OneHot)
pay_encoded = oh_encoder.transform(input_df[['preferred_payment_type']])
pay_encoded_df = pd.DataFrame(pay_encoded, columns=oh_encoder.get_feature_names_out(['preferred_payment_type']), index=input_df.index)

# Drop original categorical column and merge encoded ones
input_df = input_df.drop(columns=['preferred_payment_type'])
input_df = pd.concat([input_df, pay_encoded_df], axis=1)

# Make prediction
prediction = churn_model.predict(input_df)[0]
print("Prediction Result:", "Customer will Churn" if prediction == 1 else "Customer will Stay")
