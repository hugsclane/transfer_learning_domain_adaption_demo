import shap
import pytest
import numpy as np
import pandas as pd
from scipy import stats
import random as rd
import os
from keras.utils import plot_model
data_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'data')

def main():
    pre_pro_mod = True #serves as a way to indicate if preprocessing has been modified so as to regenerate data
    data = preprocess_df(pre_pro_mod)
    #model_something(data,'debt_consolidation','credit_card')
    #print(data['loan_purpose'].value_counts())

## Takes csv files in
def data_to_df(filename): #modify if we need more df inputs that arent data.
    '''
    Load the dataframe
    this function can only take properly formatted CSV files.
    '''
    try:
        file_path = os.path.join(data_dir_path,filename)
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("{} not found in data, or does not exist.",filename)
    except:
        Print("{} is not a csv, or is not properly formatted",filename)
    return

def preprocess_df(pre_pro_mod):
    ## Preprocessing is based on steps from https://gitlab.com/richdataco/rdc-public/rdc-ic/research/transfer-learning/ecmlpkdd2019/-/blob/master/preprocess_lending_club_files.R?ref_type=heads.
    input_file_name = 'loans_full_schema'
    '''
    When pre_pro_mod is False, preprocess_df will return previously generated data.
    '''
    if os.path.isfile(input_file_name + '_trimmed_normal_data.csv') and not pre_pro_mod:
        return data_to_df(input_file_name +'_trimmed_normal_data')
    std_trim = 10000 #seems aribtrary,:TODO ask about it?
    df = data_to_df(input_file_name + ".csv")

    '''
    Map grade to integers
    '''
    grade_mapping = {
        "A": 7,
        "B": 6,
        "C": 5,
        "D": 4,
        "E": 3,
        "F": 2,
        "G": 1
    }

    df['grade'] = df['grade'].map(lambda x: grade_mapping.get(x))

    '''
    Map sub_grade to integers
    '''
    sub_grade_mapping = {
    '1':9,
    '2':7,
    '3':5,
    '4':3,
    '5':1
    }

    df['sub_grade'] = df['sub_grade'].map(lambda x: grade_mapping.get(x[0])*10 +sub_grade_mapping.get(x[1]))

    '''
    Generate cover column
    '''
    df['cover'] = df['annual_income'] / df['loan_amount'] 


    '''
    Generate revol_util column
    '''
    df['revol_util'] = ((df['total_credit_utilized']) / df['total_credit_limit'] )


    '''
    Map emp_length to floats
    '''
    df.loc[np.isnan(df['emp_length']), 'emp_length'] = 0
    
    '''
    Map loan outcomes to binary outcomes
    '''
    df['loan_status'] = np.where(df['loan_status'] == 'Fully Paid', 1, 0)
    
    '''
    Data columns to be included in the model.
    '''
    col_ls = ['issue_month','state','total_credit_limit','term','sub_grade',\
            'revol_util','interest_rate','installment','grade','emp_length',\
            'debt_to_income','balance','total_credit_utilized','accounts_opened_24m',\
            'loan_status', 'loan_purpose','application_type','annual_income','loan_amount','cover']
    '''
    Normaliztion and standard deviation trim.
    '''
    #:TODO trim_function = lambda col:
    # normalize_function = lambda col: (col - col.min()) / \
    #     (col.max() - col.min() if col.max() - col.min() != 0 else 1)
    #     norm_df_ df[col_ls].apply(trim_function).apply(normalize_function)
    df[col_ls].to_csv(os.path.join(data_dir_path, input_file_name + '_trimmed_normal_data'+".csv"), sep = ',',mode= 'w')
    return df[col_ls]

def source_target_split(data,sName,tName,spl_ratio):
    rand_seed = 54
    '''
    Function that splits data for PSC,
    returns source and target splits.
    '''
    df_size =  len(data[data['loan_purpose'] == sName].axes[0]) if len(data[data['loan_purpose'] == sName].axes[0])< \
        len(data[data['loan_purpose'] == tName].axes[0]) else len(data[data['loan_purpose'] ==tName].axes[0]) #checks which loan_purpose is smaller then uses that as the maximum size for the sum of samples.
    s_df = data[data['loan_purpose'] == sName].sample(n = round(df_size*spl_ratio) , random_state=rand_seed)
    t_df = data[data['loan_purpose'] == tName].sample(n = round(df_size*(1-spl_ratio)),  random_state=rand_seed)
    '''
    Will fail if sampling did not occur correctly.
    '''
    assert(df_size == len(s_df.axes[0]) + len(t_df.axes[0]))
    return s_df,t_df


def model_something(data,sName,tName):
    spl_ratio = 0
    s_df,t_df = source_target_split(data,sName,tName,spl_ratio)
    return

''' BELOW ARE NOTES FOR DEVLOPMENT DOWN THE LINE, YOU CAN IGNORE THIS
use KS to quant delta source(sd) and target domain(ts).
beyond some threshold(thresh) of delta source and target domain, select those features for domain adaption
(da_feat_dict).

include other highly correlated features(cor_feat) for each selected features(key from da_feat_dict:  val from cor_feat),
populate da_feat_dict, with correlated features.
'''

#:TODO find the correct combinations by testing model accuracy (weighting correlated features or adding/removing them?)






if __name__ == "__main__":
    main()
