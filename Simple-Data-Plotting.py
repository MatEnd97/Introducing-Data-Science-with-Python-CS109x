import pandas as pd
import matplotlib.pyplot as plt


data_filename = 'Advertising.csv'
df =pd.read_csv(data_filename)
df_new = df.iloc[1:7]
x_values = df_new.iloc[:, 1]  
y_values = df_new.iloc[:, 4]  

plt.scatter(x_values,y_values)


plt.xlabel('TV budget')
plt.ylabel('Sales')


plt.title('Sales vs TV budget')
plt.show()