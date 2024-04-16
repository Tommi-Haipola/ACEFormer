import pandas as pd
import matplotlib.pyplot as plt
import myemd
import math as m
import numpy as np

normed = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))

df = pd.read_csv('./data/NDX100.csv')

y1 = df['close'].to_numpy()
y2 = df['vol'].to_numpy()
y3 = df['close_x'].to_numpy()
y4 = df['close_y'].to_numpy()
y1 = y1[2245:2385]
y2 = y2[2245:2385]
y3 = y3[2245:2385]
y4 = y4[2245:2385]

x = np.linspace(1,len(y1),len(y1))
y1 = np.array([y1]).reshape((-1,1))
y2 = np.array([y2]).reshape((-1,1))
y3 = np.array([y3]).reshape((-1,1))
y4 = np.array([y4]).reshape((-1,1))

ys = [y1,y2,y3,y4]

yk = np.concatenate([y1,y2,y3,y4],axis=1)

y = np.array([yk])

print(y.shape)
#'''
k = np.array(myemd.ACEEMD_Base(y,emd_type=2,alpha=0.8))
print(k.shape)

#l = myemd.emd(y1)
#'''

'''
plt.plot(x,y1)
plt.plot(x,l[0])
plt.plot(x,l[1])
plt.show()
#'''

#'''
for i in range(k.shape[2]):
    plt.plot(x,ys[i])
    plt.plot(x,k[0][:,i])
    plt.show()
#'''