import ast

pos = ["חיובי", "jhuch", "+", "=", "pos", "fish", "amplified"]
inter = ["inter", "בינוני", "chbubh", "equivocal", "indet", "borderline"]
neg = ["שלילי", "akhkh", "akhah", "-", "neg", "non", "not", "no", "0", "?", "pending", "o", "test"]

import re
import numpy as np
import pandas as pd
from datetime import datetime as dt

TARGET_PATH = 'train.labels.0.csv'
DATA_PATH = "train.feats.csv"

FEATURE_12_DEFAULT = -1
FEATURE_18_DEFAULT = -1
FEATURE_19_DEFAULT = -1  # TODO check with Roey if still needed if doing fillna(NONE)
FEATURE_31_DEFAULT = 0
FEATURE_32_DEFAULT = 0

COLUMNS_TO_REMOVE = [' Hospital', 'User Name', 'אבחנה-Surgery date1',
                     'אבחנה-Surgery date2', 'אבחנה-Surgery date3',
                     'אבחנה-Surgery name1', 'אבחנה-Surgery name2', 'אבחנה-Surgery name3',
                     'אבחנה-Tumor depth','אבחנה-Tumor width','surgery before or after-Actual activity']
COLUMNS_TO_DUMMIES = ['אבחנה-Basic stage','אבחנה-Histological diagnosis','אבחנה-Histopatological degree','אבחנה-Lymphatic penetration'
                      ,'אבחנה-M -metastases mark (TNM)','אבחנה-Margin Type','אבחנה-N -lymph nodes mark (TNM)',
                      'אבחנה-Side','אבחנה-Stage','אבחנה-Surgery sum','אבחנה-T -Tumor mark (TNM)']

def to_epoch(data:pd.DataFrame, cols:list) -> pd.DataFrame:
    """
    converts given cols in data to epoch time
    returns fixed data
    """
    for col in cols:
        data[col] = pd.to_datetime(data[col]).apply(lambda x:dt.timestamp(x) if type(x) ==pd.Timestamp else -1)
    return data


def clean_8(data: pd.DataFrame) -> pd.DataFrame:
    """
    as fish returns only two options (True/False), map all "pos" containing strings (heb and eng), "+/=" signs to Positive etc.
    marking data as 0 - none, 1 - intermediate, 2 - exist
    other will get 0 value
    """
    data["אבחנה-Her2"] = data["אבחנה-Her2"].apply(clean_8_apply_func)
    return data


def clean_8_apply_func(val: str) -> int:
    """
    gets singleval from feat 8 and changes it to value according to doc above
    """
    other = ['_', ')', 'Heg', 'nec', 'nrg', 'heg', 'Nag', 'nef', 'meg', 'nfg', 'ND',
             ',eg']  # got from clean_8_get_uniques

    if val is None or any(phrase in val for phrase in neg + other):
        return 0
    if any(phrase in val for phrase in pos) or val.isdigit():
        return 2
    if any(phrase in val for phrase in inter):
        return 1
    else:
        return 0


# def clean_8_get_uniques():
#   """
#   returns all 'other' values in this col, in regard to default pos/neg/inter lists
#   """
#   her2d = data["אבחנה-Her2"]
#   [name for name in her2d if
#    name is not None and type(name) != float and not any(word in name.lower() for word in pos + neg + inter)]

def clean_11(df: pd.DataFrame):
    """
    original values- array([nan, 'none', 'NO', '-', '+', 'YES', 'no', 'single focus', 'not',
       'yes', '(-)', '?']
    2 samples with the value of "single focus" were replaced with null
    after cleaning this feature, 0 is no penetration, and 1 is penetration
    printing num of samples for each value
    did not take in a concideration other values
    """
    data["אבחנה-Ivi -Lymphovascular invasion"] = data["אבחנה-Ivi -Lymphovascular invasion"].apply(clean_11_helper)
    return data

def clean_11_helper(val:str):
    pos_penetration = ["yes","+","extensive","pos","MICROPAPILLARY VARIANT"]
    neg_penetration = [None, "?", "no", "-"]
    if val is None or any(phrase in val for phrase in neg):
        return 0
    if any(phrase in val for phrase in pos):
        return 1
    else:
        return 0

    #
    # feature = df["אבחנה-Ivi -Lymphovascular invasion"]
    # for ind, value in enumerate(feature):
    #     if value is None or value == '?':
    #         feature[ind] = 0
    #     elif value == "single focus":
    #         feature[ind] = None
    #     elif type(value) is str:
    #         if value.isalpha():
    #             value = value.lower()
    #             if (value.find('n') != -1):
    #                 feature[ind] = 0
    #             elif (value.find('y') != -1):
    #                 feature[ind] = 1
    #         elif value.find('-') != -1:
    #             feature[ind] = 0
    #         elif value.find('+') != -1:
    #             feature[ind] = 1
    # df["אבחנה-Ivi -Lymphovascular invasion"] = feature
    # return df


def clean_12(df):
    df['אבחנה-KI67 protein'] = df['אבחנה-KI67 protein'].apply(_clean_12_s)
    return df


def _clean_12_s(s):
    if s is None:
        return FEATURE_12_DEFAULT
    if type(s) != str:
        return s
    s = str.lower(s)
    m = re.findall(r"\d{1,2}-\d{1,2}", s)
    if m:
        l = [int(i) for i in m[0].split('-')]
        return np.mean(l)
    if 'score' in str.lower(s):
        if '1' in s or 'i' in s:
            return 5
        elif '2' in s or 'ii' in s:
            return 10
        elif '3' in s or 'iii' in s:
            return 20
        elif '4' in s or 'iv' in s:
            return 50
        else:
            return s
    m = re.findall(r"\d{1,2}", s)
    if m:
        return max([int(i) for i in m])
    if 'h' in str.lower(s):
        return 20
    if 'l' in str.lower(s):
        return 10
    return FEATURE_12_DEFAULT


def clean_15(df: pd.DataFrame):
    """
    original train value - array([nan, 'N1', 'N0', '#NAME?', 'N1a', 'N2', 'NX', 'N1c', 'ITC',
       'N1mic', 'N3', 'N3d', 'N2a', 'Not yet Established', 'N1b', 'N3b']
    num of unique values in train data - 16 but in document is 21

    meaning of feature from the internet " N1: The cancer has spread to 1 or more lymph nodes on the same side as the primary tumor,
    and the cancer found in the node is 6 cm or smaller. N2: Cancer has spread to 1 or more lymph nodes on either side of the body,
    and none is larger than 6 cm. N3: The cancer is found in a lymph node and is larger than 6 cm."

    """
    feature = df["אבחנה-N -lymph nodes mark (TNM)"]
    for ind, value in enumerate(feature):
        if type(value)==str and (value == 'Not yet Established' or value.find("NAME") != -1):
            feature[ind] = None
    df["אבחנה-N -lymph nodes mark (TNM)"] = feature
    return df


def clean_16(df: pd.DataFrame):
    """
    Tumor mark.
    replace all strings indicating of None with 0.0
    """
    df["אבחנה-T -Tumor mark (TNM)"] = df["אבחנה-T -Tumor mark (TNM)"].replace(['Not yet Established', None], 0.0)
    return df


def clean_18(df):
    df['אבחנה-Nodes exam'] = df['אבחנה-Nodes exam'].apply(lambda x: FEATURE_18_DEFAULT if x is None else x)
    return df


def clean_19(df):
    df["אבחנה-Positive nodes"] = df["אבחנה-Positive nodes"].apply(lambda x: FEATURE_19_DEFAULT if x is None else x)
    return df


def clean_30(df: pd.DataFrame):
    """
    tumor width
    replace all None with zero
    """
    df["אבחנה-Tumor width"] = df["אבחנה-Tumor width"].replace([None], 0.0)
    return df


def _clean_31_s(s):
    if s is None:
        return FEATURE_31_DEFAULT
    if type(s) == int:
        return int(s >= 10)
    s = str.lower(s)
    m = re.findall(r"\d{1,2}", s)
    if m:
        return int(int(m[0]) >= 10)
    if '+' in s or 'p' in s or 'ח' in s or 'high' in s or 'strongly' in s:
        return 1
    if '-' in s or '_' in s or 'low' in s or 'n' in s or 'ש' in s or 'nge' in s or 'eg' in s:
        return 0
    return 0


def clean_31(df):
    df['אבחנה-er'] = df['אבחנה-er'].apply(_clean_31_s)
    return df


def _clean_32_s(s):
    if s is None:
        return FEATURE_32_DEFAULT
    if type(s) == int:
        return int(s >= 10)
    s = str.lower(s)
    m = re.findall(r"\d{1,2}", s)
    if m:
        return int(int(m[0]) >= 10)
    if '+' in s or 'p' in s or 'ח' in s or 'high' in s or 'strongly' in s:
        return 1
    if '-' in s or '_' in s or 'low' in s or 'n' in s or 'ש' in s or 'nge' in s or 'eg' in s:
        return 0
    return 0


def clean_32(df):
    df['אבחנה-pr'] = df['אבחנה-pr'].apply(_clean_32_s)
    return df


def group_by_id(df):
    """
    should be activate after Form Name became one hot + sum
    and df has target

    :param df:
    :return:
    """
    form_cols = [col for col in df.columns if 'Form Name' in col]
    rest_cols = [col for col in df.columns if 'Form Name' not in col]
    form_cols_df = pd.DataFrame(df.groupby('id-hushed_internalpatientid')[form_cols].sum())
    df = pd.DataFrame(df.groupby('id-hushed_internalpatientid')[rest_cols].max())
    df[form_cols] = form_cols_df
    return df


def form_name_to_one_hot(df):
    df = pd.get_dummies(df, columns=[' Form Name', ])
    return df


def add_target(df):
    target = pd.read_csv(TARGET_PATH)
    df['target'] = target['אבחנה-Location of distal metastases'].apply(ast.literal_eval)

    df = df.drop('target', 1).join(df.target.str.join('|').str.get_dummies())
    return df


def remove_cols(data, cols_to_remove):
    data.drop(columns=cols_to_remove, inplace=True)
    return data


def dummies(df, features=COLUMNS_TO_DUMMIES):
    df = pd.get_dummies(df, columns=features)
    return df

def manipulate_surgeries(data):
    pass


if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    data = data.fillna(np.nan).replace([np.nan], [None])
    data = remove_cols(data, COLUMNS_TO_REMOVE)
    data = to_epoch(data, cols=['surgery before or after-Activity date','אבחנה-Diagnosis date'])

    data = clean_8(data)
    data = clean_11(data)
    data = clean_12(data)
    data = clean_15(data)
    data = clean_16(data)
    data = clean_18(data)
    data = clean_19(data)
    # data = clean_30(data)
    data = clean_31(data)
    data = clean_32(data)


    # for i in range(1, 33):
    #     command = f'data = clean_{i}(data)'
    #     print(command)
    #     try:
    #         eval(command)
    #     except:
    #         pass

    data = form_name_to_one_hot(data)
    data = add_target(data)
    data = group_by_id(data)
    data.drop(columns = ['id-hushed_internalpatientid'], inplace=True)
    data = dummies(data)
    data.to_csv('clean_data.csv', index=False)
