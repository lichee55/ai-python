from sklearn.neural_network import MLPRegressor
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error

boston_dataset = datasets.load_boston()
x = boston_dataset.data
y = boston_dataset.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# 학습
classifier = MLPRegressor(hidden_layer_sizes=(
    5, 2), max_iter=10000, random_state=1)
classifier.fit(x_train, y_train)
# 테스트
predicted = classifier.predict(x_test)
mse = mean_squared_error(y_test, predicted)
rmse = np.sqrt(mse)
# 결과 출력
for i in range(len(y_test)):
    print(y_test[i], predicted[i])

print()
print("RMSE : ", rmse)
