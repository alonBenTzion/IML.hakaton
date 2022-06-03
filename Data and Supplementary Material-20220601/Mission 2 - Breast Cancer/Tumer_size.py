from preproccess import load_train_test_data_and_clean
from xgboost import XGBRegressor
from postproccessor import y_pred_to_origin_size_and_order
TARGET_NAMES = 'אבחנה-Tumor size'



TRAIN_X_PATH = 'train.feats.csv'
TRAIN_Y_PATH = 'train.labels.1.csv'
TEST_X_PATH = 'test.feats.csv'


def run_tumor_size_model():
    X, Y, x_test, x_test_origin = load_train_test_data_and_clean(TRAIN_X_PATH, TRAIN_Y_PATH, TEST_X_PATH, TARGET_NAMES)
    xgb = XGBRegressor(objective='reg:squarederror')
    xgb.fit(X, Y)
    y_pred = xgb.predict(x_test)
    y_pred = [i if i>0 else 0 for i in y_pred]
    return y_pred_to_origin_size_and_order(y_pred, x_test, x_test_origin)

if __name__ == '__main__':
    print(run_tumor_size_model())



