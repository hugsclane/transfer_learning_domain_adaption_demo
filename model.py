import shap
import pytest
import numpt as np
import pandas as pd
from scipy import stats
import random as rd
import os
from keras.utils import plot_model

def main():
    preprocess_source_dom()

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

def preprocess_source_dom(): ##:TODO Preprocessing is based on steps from https://gitlab.com/richdataco/rdc-public/rdc-ic/research/transfer-learning/ecmlpkdd2019/-/blob/master/preprocess_lending_club_files.R?ref_type=heads.
    rd.seed(54)
    df = data_to_df('loans_full_schema.csv')
    df['revol_util'] = (df['total_credit_limit'] - df['total_credit_utilized']) / df['credit_limit'] #:TODO test this to see if balance = total_credit_limit - total_credit_utilized. 
                                                                                                     #if not, check balance < total_credit_limit - total_credit_utilized so see if balance is one one loan.
    df['policy_code'] = df.apply(lambda x: 2 if ( #:TODO test this to see if 1. the lambda functions works, 2. that policy_code 2 exits in the dataset.
        df['total_credit_utilized'].isna() and
        df['total_credit_limit'].isna() and
        df['open_credit_lines'].isna() and
        df['total_credit_lines'].isna() and 
        df['grade'] in ['F', 'G'] and
        23.5 <= df['int_rate'] <= 26.06 and
        df['loan_term'] == 36 and
        df['loan_amount'] <= 15000
        ) else 1, axis=1)  
    col_ls = ['issue_month','state','total_credit_limit','term','sub_grade',\
              'revol_util','intrest_rate','installment','grade','emp_length',\
              'debt_to_income','balance','total_credit_utilized','accounts_opened_24m',\
              'loan_status', 'loan_purpose','application_type','annual_income','loan_amount']


    return df



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
