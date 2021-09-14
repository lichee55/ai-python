import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load housing data
data = datasets.load_boston()
X = data.data
y = data.target

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# 원 데이터를 이용하여 학습 및 성능 측정
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)
sv_regressor.fit(X_train, y_train)
y_test_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
print("축소 전 MSE =", round(mse, 2))

# 차원 축소
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# 축소 데이터 이용하여 학습 및 성능 측정
sv_regressor.fit(X_train_reduced, y_train)
y_test_pred_reduced = sv_regressor.predict(X_test_reduced)
mse2 = mean_squared_error(y_test, y_test_pred_reduced)
print("축소 후 MSE =", round(mse2, 2))
