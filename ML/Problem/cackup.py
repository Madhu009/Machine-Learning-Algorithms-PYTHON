

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from numpy import array

def fun(filename):
    data = pd.read_csv(filename)
    data = data.fillna(0)
    feature_cols = ['loan_amnt', 'funded_amnt', 'int_rate', 'home_ownership', 'annual_inc', 'dti',
                    'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
                    'total_rec_int', 'tot_cur_bal', 'total_rev_hi_lim']

    X = data[feature_cols]

    # encode string to int(Y label)
    encode = LabelEncoder()
    encode.fit(X['home_ownership'])
    EncodeY = encode.transform(X['home_ownership'])
    X['home_ownership'] = EncodeY

    # encode string to int(Y label)
    encode2 = LabelEncoder()
    encode2.fit(X['initial_list_status'])
    EncodeY2 = encode2.transform(X['initial_list_status'])
    X['initial_list_status'] = EncodeY2

    # Feature scale X=X/dys*hurs 365*24
    X['annual_inc'] = X['annual_inc'] / 87600

    # Diffrence of bank and investor
    X['funded_amnt'] = X['funded_amnt'] - data['funded_amnt_inv']

    # Loan amount / term
    X['loan_amnt'] = X['loan_amnt'] / 1000

    X['tot_cur_bal'] = X['tot_cur_bal'] / 100000
    X['total_rev_hi_lim'] = X['total_rev_hi_lim'] / 100000
    X['total_rec_int'] = X['total_rec_int'] / 1000
    X['revol_bal'] = X['revol_bal'] / 1000
    X['revol_util'] = X['revol_util'] / 1000
    X['total_acc'] = X['total_acc'] / 10
    X['dti']=X['dti']/10
    y = data.loan_status
    print(X)
    y=y.values
    X=X.values

    return X,y

file1="C:/Users/Madhu/Desktop/train_indessa.csv"
trainX,trainY=fun(file1)

def fun1(filename):
    data = pd.read_csv(filename)
    data = data.fillna(0)
    feature_cols = ['loan_amnt', 'funded_amnt', 'int_rate', 'home_ownership', 'annual_inc', 'dti',
                    'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
                    'total_rec_int', 'tot_cur_bal', 'total_rev_hi_lim']

    X = data[feature_cols]

    # encode string to int(Y label)
    encode = LabelEncoder()
    encode.fit(X['home_ownership'])
    EncodeY = encode.transform(X['home_ownership'])
    X['home_ownership'] = EncodeY

    # encode string to int(Y label)
    encode2 = LabelEncoder()
    encode2.fit(X['initial_list_status'])
    EncodeY2 = encode2.transform(X['initial_list_status'])
    X['initial_list_status'] = EncodeY2

    # Feature scale X=X/dys*hurs 365*24
    X['annual_inc'] = X['annual_inc'] / 8760

    # Diffrence of bank and investor
    X['funded_amnt'] = X['funded_amnt'] - data['funded_amnt_inv']

    # Loan amount / term
    X['loan_amnt'] = X['loan_amnt'] / 1000

    X['tot_cur_bal'] = X['tot_cur_bal'] / 10000
    X['total_rev_hi_lim'] = X['total_rev_hi_lim'] / 10000
    X['total_rec_int'] = X['total_rec_int'] / 1000
    X['revol_bal'] = X['revol_bal'] / 1000
    X['revol_util'] = X['revol_util'] / 1000
    X['total_acc'] = X['total_acc'] / 10
    X = X.values

    return X

file2="C:/Users/Madhu/Desktop/train_indessa.csv"
testX=fun1(file2)


