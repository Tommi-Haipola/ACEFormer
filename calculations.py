import pandas as pd
import matplotlib.pyplot as plt
import myemd
import math as m
import numpy as np

df = pd.read_csv('./data/NDX100.csv')

y1 = df['close'].to_numpy()
y1 = y1[1245:1345]
x = np.linspace(1,len(y1),len(y1))
y = np.array([np.array([y1]).reshape((-1,1))])

print(y.shape)
#'''
k = np.array(myemd.ACEEMD_Base(y,emd_type=2))

l = myemd.emd(y1)
#'''

#'''
plt.plot(x,y1)
plt.plot(x,l[0])
plt.plot(x,l[1])
plt.show()
#'''

#'''
plt.plot(x,y1)
plt.plot(x,k[0].reshape((-1,)))
plt.show()
#'''