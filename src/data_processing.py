"""
Data Processing Module for Loan Approval System

Provides functions to clean, preprocess, and validate batch data before analysis or modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

# Load environment variables and config
BASE_DIR = Path(os.getcwd())
env_path = BASE_DIR / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
config_path = BASE_DIR / 'config.yaml'
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {}

def load_data(file_path):
    """Load raw batch data from CSV."""
    return pd.read_csv(file_path, low_memory=False)

def standardize_columns(df):
    """Standardize column names to lowercase and strip whitespace."""
    df.columns = df.columns.str.lower().str.strip()
    return df

def clean_missing_values(df, threshold=0.4):
    """Drop columns with >threshold missing values, fill remaining NaNs."""
    missing_pct = df.isna().mean()
    drop_cols = missing_pct[missing_pct > threshold].index.tolist()
    df = df.drop(columns=drop_cols)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def filter_modeling_records(df):
    """Filter dataset to only include records with valid target values."""
    return df[df['target'].notna()].copy()

def preprocess_batch(file_path):
    """Full batch preprocessing pipeline."""
    df = load_data(file_path)
    df = standardize_columns(df)
    df = clean_missing_values(df)
    if 'target' in df.columns:
        df = filter_modeling_records(df)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch Data Cleaning for Loan Approval System")
    parser.add_argument("--input", required=True, help="Path to raw batch CSV file")
    parser.add_argument("--output", required=True, help="Path to save cleaned CSV file")
    args = parser.parse_args()

    print(f"Loading batch data from: {args.input}")
    df_clean = preprocess_batch(args.input)
    print(f"Saving cleaned data to: {args.output}")
    df_clean.to_csv(args.output, index=False)
    print("✅ Batch data cleaned and saved.")