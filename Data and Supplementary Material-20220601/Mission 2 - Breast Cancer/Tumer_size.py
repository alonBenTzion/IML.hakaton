# from sklearn.datasets import make_classification
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.utils import shuffle
# from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from .load_data_and_clean import load_train_test_data_and_clean

TARGET_NAMES = ['אבחנה-Tumor size']



TRAIN_X_PATH = 'train.feats.csv'
TRAIN_Y_PATH = 'train.labels.1.csv'
TEST_X_PATH = 'test.feats.csv'


if __name__ == '__main__':
    X, Y, x_test = load_train_test_data_and_clean(TRAIN_X_PATH, TRAIN_Y_PATH, TEST_X_PATH
                                                  , 'אבחנה-Tumor size')

    # Split the data into training/testing sets
    X_train = X[:-20]
    X_test = X[-20:] #NOT the x_test from above

    # Split the targets into training/testing sets
    y_train = Y[:-20]
    y_test = Y[-20:]

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    plt.scatter(X_test, y_test, color="black")
    plt.plot(X_test, y_pred, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()





    # n_samples, n_features = X.shape # 10,100
    # n_outputs = Y.shape[1] # 3
    # n_classes = 3
    #
    # # forest = RandomForestClassifier(random_state=1)
    # svm = SVC()
    # multi_target_forest = MultiOutputClassifier(svm, n_jobs=2)
    # multi_target_forest.fit(X, Y)
    # print(multi_target_forest.predict(x_test))