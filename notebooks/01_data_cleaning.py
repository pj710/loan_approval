#%%
## Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from src.utils.config_loader import load_config
from src.utils.paths import find_project_root, resolve_path

warnings.filterwarnings('ignore')

project_root = find_project_root(__file__)
os.chdir(project_root)

## set configuration variables using config.yaml
config = load_config(project_root / 'config.yaml')
data_path = config['paths']
raw_data_path = resolve_path(data_path['data_raw'], project_root)
processed_data_path = resolve_path(data_path['data_processed'], project_root)
model_data_path = resolve_path(data_path['models'], project_root)
reports_path = resolve_path(data_path['reports'], project_root)
results_path = resolve_path(data_path['results'], project_root)
# %%
# Load clean data with target
data = pd.read_csv(os.path.join(processed_data_path, 'cleaned_with_target.csv'))
data.head(5)
# %%
# To Do
# - filter for owner occupied home purchase transactions, 
# - convert income to 000's
# - fix data types 
# - Check for missing values
# - Check for duplicate rows
# - Perform basic statistical analysis
# - Visualize distributions of key variables (interest, dti, loan amount, income, property value, target, combined loan to value)

# Filter for Principal residence home purchase transactions
data = data[(data['occupancy_type'] == 'Principal residence') & (data['loan_purpose'] == 'Home purchase')]

#%%
# Fix data types
data['county_code'] = data['county_code'].astype(str).replace('.0', '', regex=False).str.zfill(4)
data['decision'] = data['decision'].astype('category')
data['debt_to_income_ratio'] = data['debt_to_income_ratio'].str.replace('.0', '', regex=False)

# Convert income to 000's
data['income'] = data['income'] * 1000

data.to_csv(os.path.join(processed_data_path, 'cleaned_data_v2.csv'), index=False)

# %%
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)
data.dropna(inplace=True, how='any', subset=None) # drop missing rows

# Check for duplicate rows
duplicate_rows = data.duplicated().sum()
print("Duplicate rows:", duplicate_rows)
data.drop_duplicates(inplace=True) # drop duplicate rows
# Save the cleaned data after removing missing values and duplicates
data.to_csv(os.path.join(processed_data_path, 'cleaned_data_v3.csv'), index=False)
# %%
# Perform basic statistical analysis
statistics_num = data.describe(include='number')
statistics_cat = data.describe(include=['category'])

print("Basic statistical analysis (numerical):\n", statistics_num)
print("\n")
print("Basic statistical analysis (categorical):\n", statistics_cat)
# %%
# Visualize distributions of key variables using subplots

key_variables = ['interest_rate', 'debt_to_income_ratio', 'loan_amount', 'income', 'property_value', 'decision', 'combined_loan_to_value_ratio', 'total_loan_costs']
fig, axes = plt.subplots(4, 2, figsize=(10, 16))
for i, var in enumerate(key_variables):
        ax = axes[i // 2, i % 2]
        if data[var].dtype.name == 'category' or data[var].dtype.name == 'object':
            data[var].value_counts().plot(kind='bar', ax=ax)
        else:
            data[var].hist(bins=50, ax=ax)
        ax.set_title(f'Distribution of {var}')
        ax.set_xlabel(var)
        ax.set_ylabel('log Frequency')
        ax.set_yscale('log')
plt.tight_layout()
plt.show()

fig.savefig(os.path.join(reports_path, 'key_variables_distributions.png'))
# %%
