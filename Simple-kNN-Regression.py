
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

#Importaciòn de Folder
filename = 'Advertising.csv'
df_adv = pd.read_csv(filename)

#mostrar los primeros 5 datos
df_adv.head()

x_true = df_adv.TV.iloc[5:13]

y_true = df_adv.Sales.iloc[5:13]

idx = np.argsort(y_true).values

x_true  = x_true.iloc[idx].values

y_true  = y_true.iloc[idx].values

def find_nearest(array,value):
    
    idx = pd.Series(np.abs(array-value)).idxmin()

    return idx, array[idx]

x = np.linspace(np.min(x_true), np.max(x_true))

y = np.zeros((len(x)))

for i, xi in enumerate(x):

    y[i] = y_true[find_nearest(x_true, xi )[0]]

plt.plot(x, y, '-.')

plt.plot(x_true, y_true, 'kx')

plt.title('TV vs Sales')
plt.xlabel('TV budget in $1000')
plt.ylabel('Sales in $1000')
plt.show()