import pandas as pd
import matplotlib.pyplot as plt
import myemd
import math as m
import numpy as np

normed = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))

df = pd.read_csv('./data/NDX100.csv')

y1 = df['close'].to_numpy()
y2 = df['vol'].to_numpy()
y1 = y1[1245:1445]
y2 = y2[1245:1445]

x = np.linspace(1,len(y1),len(y1))
y1 = np.array([y1]).reshape((-1,1))
y2 = np.array([y2]).reshape((-1,1))

yk = np.concatenate([y1,y2],axis=1)

y = np.array([yk])

print(y.shape)
#'''
k = np.array(myemd.ACEEMD_Base(y,emd_type=0))
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
plt.plot(x,y1)
plt.plot(x,k[0][:,0])
plt.show()
#'''

#'''
plt.plot(x,k[0][:,1])
plt.plot(x,y2)
plt.show()
#'''