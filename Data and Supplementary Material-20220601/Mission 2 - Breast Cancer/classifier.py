from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

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
if __name__ == '__main__':

    data= pd.read_csv('clean_data.csv')
    x_cols = [c for c in data.columns if c not in TARGET_NAMES]
    X = data[x_cols]
    Y=data[TARGET_NAMES]
    n_samples, n_features = X.shape # 10,100
    n_outputs = Y.shape[1] # 3
    n_classes = 3
    forest = RandomForestClassifier(random_state=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=2)
    multi_target_forest.fit(X, Y)
    multi_target_forest.predict(X)