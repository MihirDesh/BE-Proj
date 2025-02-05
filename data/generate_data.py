import numpy as np
import pandas as pd
import os

def generate_synthetic_data():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    product_ids = ['A', 'B', 'C', 'D', 'E']

    data = []
    for date in dates:
        for product in product_ids:
            order_quantity = np.random.randint(10, 200)
            demand = np.random.randint(30, 150)
            inventory_level = np.random.randint(100, 1000)
            lead_time = np.random.randint(1, 15)
            machine_availability = np.random.choice([0, 1], p=[0.1, 0.9])
            supplier_reliability = np.random.uniform(0.7, 1.0)

            data.append([date, product, order_quantity, demand, inventory_level, lead_time, machine_availability, supplier_reliability])

    df = pd.DataFrame(data, columns=['Date', 'Product_ID', 'Order_Quantity', 'Demand', 'Inventory_Level', 'Lead_Time', 'Machine_Availability', 'Supplier_Reliability'])

    # Ensure 'data' directory exists
    os.makedirs("data", exist_ok=True)
    
    df.to_csv('data/synthetic_data.csv', index=False)
    print("Realistic synthetic dataset generated and saved.")

# Generate synthetic data
generate_synthetic_data()
