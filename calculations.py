import pandas as pd

predict_df = pd.read_csv('./result/predict.csv',header=None)
true_df = pd.read_csv('./result/true.csv',header=None)

print(predict_df.iloc[:,4])
print(true_df)