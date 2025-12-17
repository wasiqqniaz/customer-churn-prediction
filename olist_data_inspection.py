import pandas as pd

# Show all columns when printing
pd.set_option('display.max_columns', None)

# Load datasets
customer = pd.read_csv('olist_customers_dataset.csv')
geo = pd.read_csv('olist_geolocation_dataset.csv')
item = pd.read_csv('olist_order_items_dataset.csv')
payment = pd.read_csv('olist_order_payments_dataset.csv')
review = pd.read_csv('olist_order_reviews_dataset.csv')
order = pd.read_csv('olist_orders_dataset.csv')
product = pd.read_csv('olist_products_dataset.csv')
seller = pd.read_csv('olist_sellers_dataset.csv')
product_name = pd.read_csv('product_category_name_translation.csv')

# Store in a dictionary
datasets = {
    "customer": customer,
    "geolocation": geo,
    "item": item,
    "payment": payment,
    "review": review,
    "order": order,
    "product": product,
    "seller": seller,
    "product_name": product_name
}

# Print shape and first few rows for each dataset
for name, df in datasets.items():
    print(f"\n---{name} dataset---")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print(f"Header {name}:")
    print(df.head())
