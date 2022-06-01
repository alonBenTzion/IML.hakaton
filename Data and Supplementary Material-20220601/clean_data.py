import re
import numpy as np

FEATURE_12_DEFAULT = -1
FEATURE_18_DEFAULT = -1

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

import pandas as pd
DATA_PATH = 'Mission 2 - Breast Cancer/train.feats.csv'
df = pd.read_csv(DATA_PATH)
print(sum(df['אבחנה-Nodes exam'].isna()))
print(df['אבחנה-Nodes exam'].value_counts())
# df['אבחנה-Nodes exam'] = df['אבחנה-Nodes exam'].apply(clean_18)
# [i for i in df['אבחנה-KI67 protein'].unique() if type(i)!=int]
