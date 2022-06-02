pos = ["חיובי", "jhuch","+","=","pos","fish","amplified"]
inter = ["inter", "בינוני","chbubh","equivocal","indet","borderline"]
neg = ["שלילי", "akhkh","akhah","-","neg","non", "not","no","0","?","pending","o","test"]

import re
import numpy as np
import pandas as pd

TARGET_PATH = 'Data and Supplementary Material-20220601/Mission 2 - Breast Cancer/train.labels.0.csv'
DATA_PATH = "train.feats.csv"

FEATURE_12_DEFAULT = -1
FEATURE_18_DEFAULT = -1
FEATURE_19_DEFAULT = -1 #TODO check with Roey if still needed if doing fillna(NONE)
FEATURE_31_DEFAULT = 0
FEATURE_32_DEFAULT = 0


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

def clean_12(df):
  df['אבחנה-KI67 protein'] =  df['אבחנה-KI67 protein'].apply(clean_12)

def _clean_12_s(s):
  if s is None:
    return FEATURE_12_DEFAULT
  if type(s)!=str:
    return s
  s=str.lower(s)
  m=re.findall(r"\d{1,2}-\d{1,2}", s)
  if m:
    l=[int(i) for i in m[0].split('-')]
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
  m=re.findall(r"\d{1,2}", s)
  if m:
    return max([int(i) for i in m])
  if 'h' in str.lower(s):
    return 20
  if 'l' in str.lower(s):
    return 10
  return FEATURE_12_DEFAULT

def clean_18(df):
  df['אבחנה-Nodes exam']=df['אבחנה-Nodes exam'].apply(lambda x:FEATURE_18_DEFAULT if x is None else x)

def clean_19(df):
  df['אבחנה-Nodes exam']=df['אבחנה-Nodes exam'].apply(lambda x:FEATURE_19_DEFAULT if x is None else x)

def _clean_31_s(s):
  if s is None:
    return FEATURE_31_DEFAULT
  if type(s) == int:
    return int(s>=10)
  s=str.lower(s)
  m = re.findall(r"\d{1,2}", s)
  if m:
    return int(m[0]>=10)
  if '+' in s or 'p' in s or 'ח' in s or 'high' in s or 'strongly' in s:
    return 1
  if '-' in s or '_' in s or 'low' in s or 'n' in s or 'ש' in s or 'nge' in s or 'eg' in s:
    return 0
  return 0

def clean_31(df):
  df['אבחנה-er'] =  df['אבחנה-er'].apply(_clean_31_s)

def _clean_32_s(s):
  if s is None:
    return FEATURE_32_DEFAULT
  if type(s) == int:
    return int(s>=10)
  s=str.lower(s)
  m = re.findall(r"\d{1,2}", s)
  if m:
    return int(m[0]>=10)
  if '+' in s or 'p' in s or 'ח' in s or 'high' in s or 'strongly' in s:
    return 1
  if '-' in s or '_' in s or 'low' in s or 'n' in s or 'ש' in s or 'nge' in s or 'eg' in s:
    return 0
  return 0

def clean_32(df):
  df['אבחנה-pr'] = df['אבחנה-pr'].apply(_clean_32_s)

def group_by_id(df):
  """
  should be activate after Form Name became one hot + sum
  and df has target

  :param df:
  :return:
  """
  form_cols = [col for col in df.columns if 'Form Name' in col]
  rest_cols =  [col for col in df.columns if 'Form Name' not in col]
  form_cols_df = pd.DataFrame(df.groupby('id-hushed_internalpatientid')[form_cols].sum())
  df = pd.DataFrame(df.groupby('id-hushed_internalpatientid')[rest_cols].max())
  df[form_cols]=form_cols_df
  return df

def form_name_to_one_hot(df):
  df = pd.get_dummies(df, columns=[' Form Name', ])
  return df

def add_target(df):
  target = pd.read_csv(TARGET_PATH)
  df['target'] = target['אבחנה-Location of distal metastases']
  return df

if __name__ == '__main__':
  data = pd.read_csv(DATA_PATH)
  data = data.fillna(np.nan).replace([np.nan], [None])
  for i in range(1,33):
    eval(f'data = clean_{i}(data)')
  data = form_name_to_one_hot(data)
  data = add_target(data)
  data = group_by_id(data)