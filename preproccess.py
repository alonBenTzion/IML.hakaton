import pandas as pd

train_labels = pd.read_csv("Data and Supplementary Material-20220601/Mission 2 - Breast Cancer/train.labels.0.csv")
train_samples = pd.read_csv("Data and Supplementary Material-20220601/Mission 2 - Breast Cancer/train.feats.csv")
df = pd.DataFrame(train_labels)
print(df[0].unique())
