import ast
import pickle
import pandas as pd
DATA_PATH = 'Data and Supplementary Material-20220601/Mission 2 - Breast Cancer/train.labels.0.csv'
df = pd.read_csv(DATA_PATH)
import numpy as np

def clean_target(x):
  l = ast.literal_eval(x)
  no_dup = set(l)
  l= list(no_dup)
  l.sort()
  if len(l)==0:
    l.append('')
  return l

labels_0 = [clean_target(x) for x in df['אבחנה-Location of distal metastases']]
# print(labels_0)
with open('labels_0_clean', 'wb') as f:
        pickle.dump(labels_0, f)
# for loading
# with open('labels_0_clean', 'rb') as f:
#     labels_0 = pickle.load(f)