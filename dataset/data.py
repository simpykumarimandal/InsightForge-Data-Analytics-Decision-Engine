
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)

n_customers = 300
n_rows = 2000

customer_ids = [f"CUST_{i}" for i in range(1, n_customers+1)]
products = ["Laptop", "Phone", "Shoes", "Clothing", "Watch", "Headphones"]
categories = ["Electronics", "Fashion", "Accessories"]
channels = ["Ads", "Organic", "Referral"]
devices = ["Mobile", "Desktop"]
locations = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai"]
payments = ["UPI", "Card", "COD"]

data = []

for i in range(n_rows):
    customer = random.choice(customer_ids)
    order_id = f"ORD_{i+1}"
    
    order_date = datetime.today() - timedelta(days=random.randint(1, 365))
    
    product = random.choice(products)
    category = random.choice(categories)
    
    price = random.randint(500, 50000)
    quantity = random.randint(1, 3)
    discount = random.choice([0, 5, 10, 15, 20])
    
    channel = random.choice(channels)
    device = random.choice(devices)
    location = random.choice(locations)
    payment = random.choice(payments)
    
    # Funnel behavior
    visited = 1
    added_to_cart = np.random.choice([0,1], p=[0.3, 0.7])
    purchased = 1 if added_to_cart == 1 and np.random.rand() > 0.2 else 0
    
    data.append([
        customer, order_id, order_date, product, category,
        price, quantity, discount, channel, device,
        location, payment, visited, added_to_cart, purchased
    ])

df = pd.DataFrame(data, columns=[
    "customer_id", "order_id", "order_date", "product", "category",
    "price", "quantity", "discount", "channel", "device",
    "location", "payment_method", "visited", "added_to_cart", "purchased"
])

# Create total spend
df["total_spend"] = df["price"] * df["quantity"] * (1 - df["discount"]/100)

# Create last purchase per customer
last_purchase = df[df["purchased"] == 1].groupby("customer_id")["order_date"].max()

df["last_purchase_date"] = df["customer_id"].map(last_purchase)

# Create churn label (no purchase in last 90 days)
today = datetime.today()
df["recency_days"] = (today - df["last_purchase_date"]).dt.days
df["churn"] = df["recency_days"].apply(lambda x: 1 if x > 90 else 0)

# Save dataset
df.to_csv("ecommerce_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())