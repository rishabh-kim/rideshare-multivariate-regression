import pandas as pd
import numpy as np

# Generate sample rideshare data
np.random.seed(42)
n_samples = 5000

data = {
    'distance': np.random.uniform(0.5, 30, n_samples),  # miles
    'surge_multiplier': np.random.uniform(1.0, 3.5, n_samples),
    'time_of_day': np.random.uniform(0, 24, n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples),
    'passenger_rating': np.random.uniform(3.5, 5.0, n_samples),
    'vehicle_type': np.random.choice(['UberX', 'UberXL', 'UberPlus'], n_samples),
    'num_passengers': np.random.randint(1, 7, n_samples),
    'traffic_level': np.random.choice(['low', 'medium', 'high'], n_samples),
}

df = pd.DataFrame(data)

# Calculate price based on features
base_price = 3.0
price = (base_price + 
         df['distance'] * 1.5 +
         df['surge_multiplier'] * 2.0 +
         df['passenger_rating'] * 0.5 +
         df['num_passengers'] * 0.25 +
         np.random.normal(0, 2, n_samples))  # Add noise

# Adjust for traffic
traffic_factor = {'low': 1.0, 'medium': 1.2, 'high': 1.5}
df['traffic_factor'] = df['traffic_level'].map(traffic_factor)
price = price * df['traffic_factor']

# Adjust for time of day (peak hours)
peak_hours = (df['time_of_day'] >= 7) & (df['time_of_day'] <= 9) | (df['time_of_day'] >= 17) & (df['time_of_day'] <= 19)
price = price * (1.3 * peak_hours + 1.0 * ~peak_hours)

# Ensure positive prices
price = np.maximum(price, 4.0)

df['price'] = price

# Save to CSV
df.to_csv('rideshare_kaggle.csv', index=False)
print(f"Generated {len(df)} sample rideshare records")
print(df.head())
print(f"\nDataset saved to rideshare_kaggle.csv")
