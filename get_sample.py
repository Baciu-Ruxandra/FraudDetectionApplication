import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
RAW_DATA_PATH = "data/raw/fraudTest.csv"   # or fraudTrain.csv if you prefer
OUTPUT_PATH = "data/raw/clean_sample_input.csv"

# Load dataset
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Input file not found: {RAW_DATA_PATH}")

df = pd.read_csv(RAW_DATA_PATH)

# Sanity check
if 'is_fraud' not in df.columns:
    raise ValueError("The 'is_fraud' column is missing from the dataset!")

# Stratified sampling (preserve class ratio)
sample_size = 500
df_sampled, _ = train_test_split(
    df,
    train_size=sample_size,
    stratify=df['is_fraud'],
    random_state=42
)

# Print class distribution before dropping
print(f"Class distribution in sample:\n{df_sampled['is_fraud'].value_counts(normalize=True)}")

# Drop unwanted columns
df_sampled = df_sampled.drop(["is_fraud", "Unnamed: 0"], axis=1, errors='ignore')

# Save sample
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_sampled.to_csv(OUTPUT_PATH, index=False)

print(f"\nStratified sample of {sample_size} rows saved to: {OUTPUT_PATH}")
