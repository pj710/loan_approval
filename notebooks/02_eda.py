## Exploratory Data Analysis
## This notebook contains the exploratory data analysis for the loan approval project. 
## Key objectives include understanding the distribution of key variables, identifying potential correlations with the target variable,
## and detecting any anomalies or patterns that may inform the modeling process. 
## Questions to be addressed include:
## q.1 - What is the distribution of the target variable (loan approval decision)? and how does it vary for protected attributes (e.g., race, ethnicity, sex)?
## q.2 - How are key underwritting variables (income, loan amount, property value, debt-to-income ratio) distributed?
## q.3 - Are there any notable correlations between the underwriting variables and the approval rate?
## q.4 - Are there any missing values or anomalies that need to be addressed before modeling?

# Import necessary libraries
#%%
## Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

#%%
# Load the cleaned data with target
data = pd.read_csv(os.path.join(processed_data_path, 'cleaned_data_v3.csv'))
data.info()

# %%
data = data.rename(columns={'lien status': 'lien_status', 
                            'total units': 'total_units', 
                            'construction method': 'construction_method'})


# %%
## q.0 - What is the size of the dataset?
print(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")

## q.1 - What is the distribution of approval rate across the dataset? and how does it vary for protected attributes

approval_rate_tbl = data['decision'].value_counts(normalize=True).sort_values(ascending=False).to_frame().reset_index()
approval_rate_tbl.columns = ['decision', 'approval_rate']
print(approval_rate_tbl)

protected_attributes = ['applicant_ethnicity_1', 'applicant_sex', 'applicant_race_1','applicant_age']
for attr in protected_attributes:
    print(f"\nDistribution of approval rate by {attr}:")
    approval_rate_by_attr = data.groupby(attr)['decision'] \
    .value_counts(normalize=True).unstack().fillna(0)
    print(approval_rate_by_attr)
    
# Plotting the distribution of approval rates by protected attributes
figure, axes = plt.subplots(2, 2, figsize=(16, 12))
for i, attr in enumerate(protected_attributes):
    ax = axes[i//2, i%2]
    approval_rate_by_attr = data.groupby(attr)['decision']\
        .value_counts(normalize=True).unstack().fillna(0)
    approval_rate_by_attr.plot(kind='bar', stacked=False, ax=ax, color=['skyblue', 'salmon'])
    ax.set_title(f'Approval Rate by {attr}', fontsize=12, fontweight='bold')
    ax.set_xlabel(attr, fontsize=10)
    ax.set_ylabel(f'Approval Rate', fontsize=10)
    
    # Abbreviate long labels
    labels = [label.get_text()[:20] + '...' if len(label.get_text()) > 20 else label.get_text() 
              for label in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(title='Decision', loc='upper right', fontsize=9)
    
plt.tight_layout()
plt.show()  

# %%
## q.2 - How are key underwritting variables (income, loan amount, property value, debt-to-income ratio, interest rate) distributed?
underwriting_vars = ['income', 'loan_amount', 'property_value', 'debt_to_income_ratio', 'interest_rate']
data[underwriting_vars].describe() 

## Plotting the distribution of underwriting variables colored by approval decision
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for i, var in enumerate(underwriting_vars):
    ax = axes[i//2, i%2]
    if var != 'debt_to_income_ratio':
        np.log1p(data[var]).plot(kind='hist', bins=30, ax=ax, color='skyblue', edgecolor='black', hue=data['decision'])
        ax.set_ylabel('Approval Rate', fontsize=10)
    else:
        # debt_to_income_ratio is categorical, so use value_counts
        data[var].value_counts(normalize=True).sort_index().plot(kind='bar', ax=ax, color='salmon', edgecolor='black', hue=data['decision'])
        ax.set_ylabel('Approval Rate', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
    ax.set_title(f'Distribution of {var}', fontsize=12, fontweight='bold')
    ax.set_xlabel(var, fontsize=10)
plt.tight_layout()
plt.suptitle('Distribution of Underwriting Variables', fontsize=14, fontweight='bold')
plt.show()
# %%
# q.3 - Are there any notable correlations between the underwriting variables and the approval rate?
underwriting_vars = [var for var in underwriting_vars if var not in ['debt_to_income_ratio']]
correlation_matrix = data[underwriting_vars + ['decision']].corr()
print(correlation_matrix['decision'].sort_values(ascending=False))

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Underwriting Variables and Approval Decision', fontsize=14, fontweight='bold')
plt.show()

# %%
# q.4 - Are there any missing values or anomalies that need to be addressed before modeling?

missing_values = data.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])   
# %%

# q.5 - Are there any anomalies in the underwriting variables that need to be addressed before modeling?
# Checking for anomalies in underwriting variables
for var in underwriting_vars:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[var])
    plt.title(f'Boxplot of {var}', fontsize=14, fontweight='bold')
    plt.xlabel(var, fontsize=12)
    plt.show()
    
# %%
# q.6 - What is the rate of withdrawn applications and how does it vary across protected attributes?
data['action_taken'].value_counts(normalize=True).sort_values(ascending=False).to_frame().reset_index()