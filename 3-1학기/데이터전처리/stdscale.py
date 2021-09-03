from sklearn.preprocessing import StandardScaler
import numpy as np

my_data = np.array([10, 3, 56, 567, 30],dtype="float").reshape(-1, 1)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(my_data)
print("평균", " : ", scaler.mean_)
print("편차", " : ", scaler.var_)
scaled_my_data = scaler.transform(my_data)
print(scaled_my_data)
