import pandas as pd

def metastases_target_format(y_pred, target_labels):
    return [str([target_labels[i] for i in range(len(row)) if row[i] == 1]) for row in y_pred]

def y_pred_to_origin_size_and_order(y_pred,x_test_clean,x_test, target_name):
    y_pred = pd.DataFrame(y_pred,index = x_test_clean.index,columns = [target_name])
    return pd.DataFrame(x_test.join(y_pred,how='left')[target_name])