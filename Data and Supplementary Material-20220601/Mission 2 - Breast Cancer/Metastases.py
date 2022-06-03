from sklearn.multioutput import MultiOutputClassifier
from preproccess import load_train_test_data_and_clean
from sklearn.tree import DecisionTreeClassifier
from postproccessor import metastases_target_format, y_pred_to_origin_size_and_order

TARGET_NAME = 'אבחנה-Location of distal metastases'
TRAIN_X_PATH = 'train.feats.csv'
TRAIN_Y_PATH = 'train.labels.0.csv'
TEST_X_PATH = 'test.feats.csv'


def run_metastases_model():
    X, Y, x_test, x_test_origin = load_train_test_data_and_clean(TRAIN_X_PATH, TRAIN_Y_PATH, TEST_X_PATH
                                                                 ,TARGET_NAME)

    tree = DecisionTreeClassifier()
    multi_target_forest = MultiOutputClassifier(tree, n_jobs=2)
    multi_target_forest.fit(X, Y)
    y_pred = multi_target_forest.predict(x_test)
    y_pred = metastases_target_format(y_pred, Y.columns)
    return y_pred_to_origin_size_and_order(y_pred, x_test, x_test_origin, TARGET_NAME)

if __name__ == '__main__':
    pred = run_metastases_model()
    pred.to_csv('pred_m.csv', index=False)
