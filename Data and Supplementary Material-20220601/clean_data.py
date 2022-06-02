import re
import numpy as np

FEATURE_12_DEFAULT = -1
FEATURE_18_DEFAULT = -1
FEATURE_31_DEFAULT = 0
FEATURE_32_DEFAULT = 0

def clean_12(df):
  df['אבחנה-KI67 protein'] =  df['אבחנה-KI67 protein'].apply(clean_18)

def _clean_12_s(s):
  if s is np.nan:
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
  df['אבחנה-Nodes exam'].fillna(FEATURE_18_DEFAULT)

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

def clean_31(df):
  df['אבחנה-er'] =  df['אבחנה-er'].apply(_clean_31_s)

def clean_32(df):
  df['אבחנה-pr'] = df['אבחנה-pr'].apply(_clean_32_s)

import pandas as pd
col_name = 'אבחנה-T -Tumor mark (TNM)'
DATA_PATH = 'Mission 2 - Breast Cancer/train.feats.csv'
df = pd.read_csv(DATA_PATH)

df.groupby(['Id-hushed_internalpatientid',''])
print(sum(df[col_name].isna()))
print(df[col_name].value_counts())
# df['אבחנה-Nodes exam'] = df['אבחנה-Nodes exam'].apply(clean_18)
# [i for i in df['אבחנה-KI67 protein'].unique() if type(i)!=int]

#16
