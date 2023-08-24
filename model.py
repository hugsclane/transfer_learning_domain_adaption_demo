import shap
import pytest
import numpy as np
import pandas as pd
from scipy import stats
import random as rd
import os
from keras.utils import plot_model

def main():
    data = preprocess_df()
    model_something(data,'debt_consolidation','credit_card')
    #print(data['loan_purpose'].value_counts())

## Takes csv files in
def data_to_df(filename): #modify if we need more df inputs that arent data.
    try:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'data',filename)
        df = pd.read_csv(dir_path)
    except FileNotFoundError:
        print("{} not found in data, or does not exist.",filename)
    except:
        Print("{} is not a csv",filename)
    return df

def preprocess_df():
    ## Preprocessing is based on steps from https://gitlab.com/richdataco/rdc-public/rdc-ic/research/transfer-learning/ecmlpkdd2019/-/blob/master/preprocess_lending_club_files.R?ref_type=heads.
    df = data_to_df('loans_full_schema.csv')
    df['revol_util'] = ((df['total_credit_utilized']) / df['total_credit_limit'] )
    df['policy_code'] = df.apply(lambda x: 2 if (
        x['total_credit_utilized']== np.NaN and
        x['total_credit_limit']== np.NaN and
        x['open_credit_lines']== np.NaN and
        x['total_credit_lines']== np.NaN and
        x['grade'] in ['F', 'G'] and
        23.5 <= x['int_rate'] <= 26.06 and
        x['loan_term'] == 36 and
        x['loan_amount'] <= 15000
        ) else 1, axis=1)

    ##checking data['policy_code'].unique() there are no policy code 2 borrowers in the dataset
    col_ls = ['issue_month','state','total_credit_limit','term','sub_grade',\
              'revol_util','interest_rate','installment','grade','emp_length',\
              'debt_to_income','balance','total_credit_utilized','accounts_opened_24m',\
              'loan_status', 'loan_purpose','application_type','annual_income','loan_amount']
    return df[col_ls]

def model_something(data,sName,tName):
    spl_ratio = 0.75
    s_df,t_df = source_target_split(data,sName,tName,spl_ratio)
    return

def source_target_split(data,sName,tName,spl_ratio):
    df_size =  len(data[data['loan_purpose'] == sName].axes[0]) if len(data[data['loan_purpose'] == sName].axes[0])< \
        len(data[data['loan_purpose'] == tName].axes[0]) else len(data[data['loan_purpose'] ==tName].axes[0]) #checks which loan_purpose is smaller then uses that as the maximum size for the sum of samples.
    s_df = data[data['loan_purpose'] == sName].sample(n = round(df_size*spl_ratio) , random_state=54)
    #outcome are, n = df_size, n =
    t_df = data[data['loan_purpose'] == tName].sample(n = round(df_size*(1-spl_ratio)),  random_state=54)
    assert(df_size == len(s_df.axes[0]) + len(t_df.axes[0]))
    return s_df,t_df



'''
use KS to quant delta source(sd) and target domain(ts).
beyond some threshold(thresh) of delta source and target domain, select those features for domain adaption
(da_feat_dict).

include other highly correlated features(cor_feat) for each selected features(key from da_feat_dict:  val from cor_feat),
populate da_feat_dict, with correlated features.
''' 

#:TODO find the correct combinations by testing model accuracy (weighting correlated features or adding/removing them?)






if __name__ == "__main__":
    main()
