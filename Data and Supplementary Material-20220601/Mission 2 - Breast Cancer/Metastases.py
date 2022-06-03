from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from preproccess import load_train_test_data_and_clean
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier

TRAIN_X_PATH = 'train.feats.csv'
TRAIN_Y_PATH = 'train.labels.0.csv'
TEST_X_PATH = 'test.feats.csv'

if __name__ == '__main__':
    X, Y, x_test = load_train_test_data_and_clean(TRAIN_X_PATH, TRAIN_Y_PATH, TEST_X_PATH
                                                  ,'אבחנה-Location of distal metastases')

    tree = DecisionTreeClassifier()
    multi_target_forest = MultiOutputClassifier(tree, n_jobs=2)
    multi_target_forest.fit(X, Y)
    print(multi_target_forest.predict(x_test))
