import pandas as pd
import numpy as np

hebrew_pos = ["חיובי", "jhuch"]
hebrew_neg = ["שלילי", "akhkh"]

import re
import numpy as np

FEATURE_12_DEFAULT = -1

def clean_12(s):
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



