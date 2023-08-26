from model import data_to_df
import numpy as np
import pandas as pd
def main():
    '''
    Exploring data without preprocessing
    '''
    data_no_pp = data_to_df("loans_full_schema.csv")
    # check_policy_code(data_no_pp)
    # unique_values_in_(data_no_pp)
    # print(data_no_pp['grade'].isna().sum())
    # print(data_no_pp['sub_grade'].isna().sum())
    print(data_no_pp['term'].unique())
    #print(data_no_pp.dtypes)
    '''
    Exploring data with preprocessing
    '''
    # data_with_pp= data_to_df("loans_full_schema_trimmed_normal_data.csv")
    # unique_values_in_(data_with_pp)
    # print(data_with_pp.dtypes)

'''
Check data for policy_code 2 values.
'''
def check_policy_code(data):
    ##checking data['policy_code'].unique() there are no policy code 2 borrowers in the dataset
    data['policy_code'] = data.apply(lambda x: 2 if (
        x['total_credit_utilized']== np.NaN and
        x['total_credit_limit']== np.NaN and
        x['open_credit_lines']== np.NaN and
        x['total_credit_lines']== np.NaN and
        x['grade'] in ['F', 'G'] and
        23.5 <= x['int_rate'] <= 26.06 and
        x['loan_term'] == 36 and
        x['loan_amount'] <= 15000
        ) else 1, axis=1)
    print(data['policy_code'].unique())

def unique_values_in_(data):
    columns_to_print = ['grade', 'sub_grade', 'interest_rate', 'installment', 'emp_length', 'debt_to_income', 'balance', 'total_credit_utilized', 'accounts_opened_24m', 'annual_income', 'loan_status']
    for column in columns_to_print:
        unique_values = data[column].unique()
        print(f"Unique datapoints in {column}:\n", unique_values)

if __name__ =="__main__":
    main()
