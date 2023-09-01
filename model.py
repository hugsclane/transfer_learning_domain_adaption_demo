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
    pre_pro_mod = False #serves as a way to indicate if preprocessing has been modified so as to regenerate data
    data = preprocess_df(pre_pro_mod)
    transfer_model(data,'debt_consolidation','credit_card')
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
    df = data_to_df(input_file_name + ".csv")
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
    df_fil = df[df['application_type'] == 'individual']
    col_ls = ['total_credit_limit','term','sub_grade',\
            'revol_util','interest_rate','installment','grade','emp_length',\
            'debt_to_income','balance','total_credit_utilized','accounts_opened_24m',\
            'loan_status','annual_income','loan_amount','cover']
    '''
    Drop non-numeric data filter out any non-individual applicants.
    '''
    df_num = df.drop(['issue_month','state'],axis=1)
    df_num = df_num[(df_num['application_type'] == 'individual')]
    df_num = df_num.drop(['application_type'],axis = 1)
    '''
    Normaliztion and standard deviation trim.
    '''

    trim_function = lambda col: col[np.abs(col - col.mean()) <= (std_trim * col.std())]
    normalize_function = lambda col: (col - col.min()) / (col.max() - col.min() if col.max() - col.min() != 0 else 1)
    norm_df = df_num[col_ls].apply(trim_function).apply(normalize_function)
    norm_df = norm_df[col_ls] 
    norm_df['loan_purpose'] = df['loan_purpose']
    norm_df.to_csv(os.path.join(data_dir_path, input_file_name + '_trimmed_normal_data'+".csv"), sep = ',',mode= 'w')
    return norm_df

def source_target_split(data,sName,tName,spl_ratio):
    rand_seed = 54
    '''
    Function that splits data for PSC,
    returns source and target splits.
    '''
    df_size =  len(data[data['loan_purpose'] == sName].axes[0]) if len(data[data['loan_purpose'] == sName].axes[0])< \
    #checks which loan_purpose is smaller then uses that as the maximum size for the sum of samples.
    len(data[data['loan_purpose'] == tName].axes[0]) else len(data[data['loan_purpose'] ==tName].axes[0]) 
    s_df = data[data['loan_purpose'] == sName].sample(n = round(df_size*spl_ratio) , random_state=rand_seed)
    t_df = data[data['loan_purpose'] == tName].sample(n = round(df_size*(1-spl_ratio)),  random_state=rand_seed)

    s_df,t_dt = s_df.drop("loan_purpose",axis=1),t_df.drop("loan_purpose",axis=1)
    '''
    Will fail if sampling did not occur correctly.
    '''
    assert(df_size == len(s_df.axes[0]) + len(t_df.axes[0]))

    return s_df,t_df

def transfer_model(data,sName,tName):
    #for generating models with different network configs
    spl_ratio_ls = [1.0,0.75,0.71,0.6,0.46,0]
    spl_ratio = 1 #temp ratio to check initial model.
    #gini(score,result)
    # Gini = g(test(M_e,s_e)
    # M_e = train(M_0,P_e,t_e,F_e)
    #:TODO develop M_0,
    #:TODO tune hyperparemeters
    #:TODO derive features from t_e
    #:TODO split M_e into fixed and free segments
    #:TODO train M_free_e => M_free_n
    #:TODO use combine to merge M_fixed_e and M_free_n into M_tranfser
    # Call Gini from M_tranfser and domain test data.
    s_df,t_df = source_target_split(data,sName,tName,spl_ratio)
    return
'''
This model is a slightly refactored generic linear model based on Andrew Voumard's model_generic :
function found in :
https://gitlab.com/richdataco/rdc-public/rdc-ic/research/transfer-learning/ecmlpkdd2019/-/blob/master/allcommon.py
'''
def linear_model(n_inputs,n_outputs,n_layers,n_nodes):
    #total layers is n_layers + an input layer and an output layer
    total_layers = n_layers + 2

    model_layers = [Input(shape=(n_inputs,))] + \
        [Dense(n_nodes, activation='relu')(model_layers[i - 1])\
         for i in range(1, n_layers + 1)]
    output = Dense(n_nodes, activation = 'sigmoid')(model_layers[-1])
    model = Model(inputs = model_layer[0], outputs = output)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def gini(score,result):
    # count obsevations and positive outcomes
    n, n_pos = len(score), sum(result)

    # Combine the scores and outcomes, then sort the combined data by score in descending order
    combined_data = list(zip(fscore, foutcome))
    combined_data.sort(key=lambda x: x[0], reverse=True)

    # Create a list of sequence numbers and initialize the cumulative outcome counter
    seq = list(range(1, n + 1))
    cum_outcome = 0

    # Calculates the number of observations with lower scores that have positive outcomes,
    # and updates cumulative outcomes. :NOTE this will only work in python 3.8+
    above = [(seq.pop(0) - (cum_outcome := cum_outcome + outcome)) for _, outcome in combined_data]

    # Calculates contibution of some observation to the gini calculation.
    area = [abv * outcome for abv, (_, outcome) in zip(above, combined_data)]

    '''
    The Gini coefficient is calculated using the intermediate values area and above
    to find the area under the curve to measure the inequality
    in the distribution of outcomes across the observations.
    '''

    auc = 1 - sum(area) / (n_pos * (n - n_pos))
    gini = auc * 2 - 1
    #:TODO test gini function.
    return gini

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
