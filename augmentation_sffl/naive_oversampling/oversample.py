import pandas as pd
import numpy as np
import os

# --- Config ---
project = 'activemq'
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # one level up from naive_oversampling
data_path = os.path.join(BASE_DIR, 'data', project, 'ground_truth.csv')
output_path = os.path.join(BASE_DIR, 'data', project, 'ground_truth_balanced.csv')

# --- Load data ---
print(f"[→] Loading original dataset from: {data_path}")
df = pd.read_csv(data_path)

# --- Check class distribution ---
smelly_df = df[df['label'] == 1]
normal_df = df[df['label'] == 0]

print("\n[Before Oversampling]")
print(f"  Smelly samples (label=1): {len(smelly_df)}")
print(f"  Normal samples (label=0): {len(normal_df)}")
print(f"  Imbalance ratio (smelly/normal): {len(smelly_df) / len(normal_df):.4f}")

# --- Oversample the smelly methods ---
smelly_oversampled = smelly_df.sample(len(normal_df), replace=True, random_state=42)

# --- Combine and shuffle ---
balanced_df = pd.concat([normal_df, smelly_oversampled])
balanced_df = balanced_df.sample(frac=1.0, random_state=42)

# --- Report after oversampling ---
print("\n[After Oversampling]")
print(f"  Total samples: {len(balanced_df)}")
print("  Label distribution:")
print(balanced_df['label'].value_counts())

# --- Save the result ---
balanced_df.to_csv(output_path, index=False)
print(f"\n[✔] Balanced dataset saved successfully:")
print(f"      Path: {output_path}")
