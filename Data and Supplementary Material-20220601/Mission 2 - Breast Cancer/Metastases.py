from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from .load_data_and_clean import load_train_test_data_and_clean

TRAIN_X_PATH = 'train.feats.csv'
TRAIN_Y_PATH = 'train.labels.0.csv'
TEST_X_PATH = 'test.feats.csv'

if __name__ == '__main__':
    X, Y, x_test = load_train_test_data_and_clean(TRAIN_X_PATH, TRAIN_Y_PATH, TEST_X_PATH
                                                  ,'אבחנה-Location of distal metastases')

    n_samples, n_features = X.shape  # 10,100
    n_outputs = Y.shape[1]  # 3
    n_classes = 3

    # forest = RandomForestClassifier(random_state=1)
    svm = SVC()
    multi_target_forest = MultiOutputClassifier(svm, n_jobs=2)
    multi_target_forest.fit(X, Y)
    print(multi_target_forest.predict(x_test))
