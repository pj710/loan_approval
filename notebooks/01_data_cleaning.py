#%%
## Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import warnings

warnings.filterwarnings('ignore')

## set working directory
if not os.getcwd().endswith('loan_approval'):
    os.chdir('/Users/josiahgordor/Desktop/DSPortfolio/Projects/loan_approval') 
    
## set configuration variables using config.yaml 
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
    data_path = config['paths']
    raw_data_path = data_path['data_raw']
    processed_data_path = data_path['data_processed']
    model_data_path = data_path['models']
    reports_path = data_path['reports']
    results_path = data_path['results']
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
data['county_code'] = data['county_code'].str.replace('.0', '', regex=False).str.zfill(4)
data['decision'] = data['decision'].astype('category')


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
statistics = data.describe()
print("Basic statistical analysis:\n", statistics)
# %%
# Visualize distributions of key variables using subplots
key_variables = ['interest_rate', 'debt_to_income_ratio', 'loan_amount', 'income', 'property_value', 'decision', 'combined_loan_to_value_ratio', 'total_loan_costs']
fig, axes = plt.subplots(4, 2, figsize=(10, 16))
for i, var in enumerate(key_variables):
        ax = axes[i // 2, i % 2]
        if data[var].dtype.name == 'category' or data[var].dtype.name == 'object':
            data[var].value_counts().plot(kind='bar', ax=ax)
        else:
            data[var].hist(ax=ax,)
        ax.set_title(f'Distribution of {var}')
        ax.set_xlabel(var)
        ax.set_ylabel('log Frequency')
        ax.set_yscale('log')
plt.tight_layout()
plt.show()

fig.savefig(os.path.join(reports_path, 'key_variables_distributions.png'))
# %%
