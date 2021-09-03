import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def create_learning_data(file_name):
    f = open(file_name, "w")
    f.write("x,y,z\n")

    np.random.seed(30) 
    x = 50 * np.random.rand(100) 
    y = 50 * np.random.rand(100)
    z = 3*x + 2*y + 4 + np.random.randn(100)

    for v1, v2, v3 in zip(x, y, z):
	    f.write(str(v1) + "," + str(v2) + "," + str(v3) + "\n")
    f.close()

create_learning_data('./machinelearning/testcase/data.csv')
my_data = pd.read_csv('./machinelearning/testcase/data.csv')
heads = my_data.head()
print(heads)

descs=my_data.describe()
print(descs)

my_data.plot(kind="scatter", x ='x', y='z') 
plt.show()
