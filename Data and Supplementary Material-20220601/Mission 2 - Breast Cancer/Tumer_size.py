from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from .Cleaner import clean_data, TUMOR_SIZE_CLEAN_CSV_NAME, DATA_PATH

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

    data= pd.read_csv(TUMOR_SIZE_CLEAN_CSV_NAME)
    x_test = clean_data(pd.read_csv(DATA_PATH))
    x_cols = [c for c in data.columns if c not in TARGET_NAMES]
    X = data[x_cols]
    Y=data[TARGET_NAMES]

    dummy_cols_to_add = set(X.columns) - set(x_test.columns)
    for col in dummy_cols_to_add:
        x_test[col] = np.zeros(len(x_test))

    dummy_cols_to_add = set(x_test.columns) - set(X.columns)
    for col in dummy_cols_to_add:
        X[col] = np.zeros(len(X))

    n_samples, n_features = X.shape # 10,100
    n_outputs = Y.shape[1] # 3
    n_classes = 3

    # forest = RandomForestClassifier(random_state=1)
    svm = SVC()
    multi_target_forest = MultiOutputClassifier(svm, n_jobs=2)
    multi_target_forest.fit(X, Y)
    print(multi_target_forest.predict(x_test))