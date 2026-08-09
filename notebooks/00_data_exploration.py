#%%
## importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from pathlib import Path

import data_profiling

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
## load raw data -----
raw_data = pd.read_csv(raw_data_path)
raw_data.columns = [col.lower() for col in raw_data.columns]
print(raw_data.head(5))
print(raw_data.info())


#%%
## drop unnecessary columns from the raw data
cols_to_drop = cols_to_drop = [
    'unnamed: 0',
    # Administrative/Identifiers
    'activity_year', 'lei', 'derived_msa_md', 'census_tract', 
    'conforming_loan_limit', 'submission_of_application', 'preapproval','aus_1','aus_2',
    'aus_3', 'aus_4', 'aus_5', 'total_units', 'construction_method' # Added total_units to the list
    
    # Sparse/Optional
    'applicant_credit_score_type', 'co_applicant_credit_score_type',
    'rate_spread', 'hoepa_status',
    
    # Denial reasons (outcome variable)
    'denial_reason_1', 'denial_reason_2', 'denial_reason_3', 'denial_reason_4',
    
    # Additional race/ethnicity fields (keep only _1 versions)
    'applicant_race_2', 'applicant_race_3', 'applicant_race_4', 'applicant_race_5',
    'co_applicant_race_1', 'co_applicant_race_2', 'co_applicant_race_3', 
    'co_applicant_race_4', 'co_applicant_race_5',
    'applicant_ethnicity_2', 'applicant_ethnicity_3', 'applicant_ethnicity_4', 'applicant_ethnicity_5',
    'co_applicant_ethnicity_1', 'co_applicant_ethnicity_2', 'co_applicant_ethnicity_3', 'derived_ethnicity',
    'derived_race','derived_sex','derived_age', 'applicant_ethnicity_observed','co_applicant_ethnicity_observed',
    'applicant_sex_observed', 'co_applicant_sex_observed', 'applicant_age_above_62', 'co_applicant_age_above_62','initially_payable_to_institution',
    
    
    # Co-applicant demographics
    'co_applicant_sex', 'co_applicant_age', 'co_applicant_ethnicity_4', 'co_applicant_ethnicity_5',
    
    # Geographic details (too granular)
    'tract_population', 'tract_minority_population_percent',
    'ffiec_msa_md_median_family_income', 'tract_to_msa_income_percentage',
    'tract_owner_occupied_units', 'tract_one_to_four_family_homes',
    'tract_median_age_of_housing_units',
    
    # Out-of-scope loan types
    'reverse_mortgage', 'open_end_line_of_credit', 'business_or_commercial_purpose', 'construction_method',
    'construction_method','purchaser_type',
    
    # Detailed loan costs (unless engineering features)
    'total_points_and_fees', 'origination_charges', 'discount_points',
    'lender_credits', 'prepayment_penalty_term', 'intro_rate_period',
    'balloon_payment', 'interest_only_payment', 'negative_amortization',
    'other_nonamortizing_features', 'manufactured_home_secured_property_type',
    'manufactured_home_land_property_interest', 'multifamily_affordable_units',
]

cols_to_drop = [col for col in cols_to_drop if col in raw_data.columns]
raw_data.drop(columns=cols_to_drop, inplace=True)

clean_data_v1  = raw_data.copy() 

## save the cleaned data to the processed data path
clean_data_v1.to_csv(processed_data_path + '/clean_data_v1.csv', index=False)
#%%
data = pd.read_csv(os.path.join(processed_data_path, 'clean_data_v1.csv'))

# set target column
action_taken = {
                '1': 'Loan originated', 
                '2': 'Application approved but not accepted', 
                '3': 'Application denied', 
                '4': 'Application withdrawn', 
                '5': 'File closed for incompleteness',
                '6': 'Purchased loan',
                '7': 'Preapproval request denied',
                '8': 'Preapproval request approved but not accepted'
                }

# filter the data to include only applications that were approved or denied
data = data[data['action_taken'].isin([action_taken['1'], action_taken['2'], action_taken['3']])].copy()

# create a binary target column indicating whether the application was approved or denied
data['decision'] = data['action_taken'].apply(lambda x: 1 if x in ['Loan originated', 'Application approved but not accepted'] else 0)

# check the distribution of the binary target column
print(data['decision'].value_counts())
print(data['decision'].value_counts(normalize=True))

data.drop(columns=['action_taken'], inplace=True)

# Save the data with target column
data.to_csv(os.path.join(processed_data_path, 'cleaned_with_target.csv'), index=False)

#%%
## generate data profile report
profile = data_profiling.ProfileReport(data, title="Data Profile Report")
profile.to_file(os.path.join(reports_path, 'data_profile_report.html'))
# %%
