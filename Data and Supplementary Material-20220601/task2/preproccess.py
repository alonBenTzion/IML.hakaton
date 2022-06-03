import pandas as pd
from Cleaner import clean_data
import numpy as np


TARGET_NAMES = ['ADR - Adrenals',
 'BON - Bones',
 'BRA - Brain',
 'HEP - Hepatic',
 'LYM - Lymph nodes',
 'MAR - Bone Marrow',
 'OTH - Other',
 'PER - Peritoneum',
 'PLE - Pleura',
 'PUL - Pulmonary',
 'SKI - Skin']

def load_and_clean_train_set(train_x_path,train_y_path, target_name):
    data = pd.read_csv(train_x_path)
    target = pd.read_csv(train_y_path)
    clean_train = clean_data(data, target, target_name)
    if target_name ==  'אבחנה-Location of distal metastases':
        x_cols = [c for c in clean_train.columns if c not in TARGET_NAMES]
        return clean_train[x_cols], clean_train[TARGET_NAMES]
    x_cols = [c for c in clean_train.columns if c != target_name]
    return clean_train[x_cols], clean_train[target_name]


def load_and_clean_test_set(test_x_path):
    data = pd.read_csv(test_x_path)
    return clean_data(data), data

def _resolve_conflicts(df1,df2):
    dummy_cols_to_add = set(df1.columns) - set(df2.columns)
    for col in dummy_cols_to_add:
        df2[col] = np.zeros(len(df2))

def resolve_dummies_conflicts(X_train, X_test):
    _resolve_conflicts(X_train,X_test)
    _resolve_conflicts(X_test,X_train)

def load_train_test_data_and_clean(train_x_path, train_y_path, test_x_path, target_name):
    X_train,Y_train = load_and_clean_train_set(train_x_path,train_y_path, target_name)
    X_test, X_tets_origin = load_and_clean_test_set(test_x_path)
    X_tets_origin.set_index('id-hushed_internalpatientid', inplace=True)
    resolve_dummies_conflicts(X_train,X_test)
    return X_train,Y_train, X_test, X_tets_origin
