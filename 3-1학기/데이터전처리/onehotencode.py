import sklearn.preprocessing as prp
import numpy as np

encoder = prp.LabelEncoder()
data = ["a","a","b","c","c"]
encoded_data = encoder.fit_transform(data)
encoder = prp.OneHotEncoder()
onehot_encoded_data = encoder.fit_transform(encoded_data.reshape(-1,1))
print(onehot_encoded_data.toarray())
