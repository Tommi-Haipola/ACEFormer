import pandas as pd
import matplotlib.pyplot as plt

predict_df = pd.read_csv('./result/predict.csv',header=None)
true_df = pd.read_csv('./result/true.csv',header=None)


pred = predict_df.iloc[:,0].to_numpy()
idx = true_df.index
true = true_df.to_numpy()
print(predict_df.iloc[:,0])
print(true_df)

plt.plot(idx,pred)
plt.plot(idx,true)
plt.show()

