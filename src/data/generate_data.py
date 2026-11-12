import pandas as pd
import numpy as np
import os

def generate_user_features(num_users=10000):
    np.random.seed(42)
    
    # 1. Dense Feature: Age
    # We generate a normal distribution around age 25, clip it between 18 and 50
    raw_ages = np.random.normal(loc=25, scale=5, size=num_users)
    ages = np.clip(raw_ages, 18, 50)
    
    # Min-Max Normalization: Neural networks fail if continuous inputs aren't scaled between 0 and 1
    ages_normalized = (ages - 18) / (50 - 18)
    
    # 2. Sparse Feature: City
    # We assign every user to one of 10 hypothetical cities (IDs 0 through 9)
    cities = np.random.randint(0, 10, size=num_users)
    
    # Construct the feature registry
    users_df = pd.DataFrame({
        'user_id': range(num_users),
        'age_norm': ages_normalized.astype(np.float32),
        'city_id': cities
    })
    
    os.makedirs("data/raw", exist_ok=True)
    users_df.to_csv("data/raw/users_features.csv", index=False)
    print(f"Successfully generated features for {num_users} users.")
    print(users_df.head())

if __name__ == "__main__":
    generate_user_features()