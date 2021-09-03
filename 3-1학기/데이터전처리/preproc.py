import pandas as pd
from sklearn import datasets

my_data = datasets.load_iris()
your_data = pd.DataFrame(my_data.data)
median = your_data[0].median()
your_data = your_data[0].fillna(median) 
print(my_data)
print(your_data)