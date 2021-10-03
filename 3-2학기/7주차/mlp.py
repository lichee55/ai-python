from sklearn.neural_network import MLPClassifier
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

iris_dataset = datasets.load_iris()
x = iris_dataset.data
y = iris_dataset.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)
classifier = MLPClassifier(hidden_layer_sizes=(
    5, 2), random_state=3, max_iter=10000)

classifier.fit(x_train, y_train)
predicted = classifier.predict(x_test)

print(predicted)
print(y_test)
print(accuracy_score(y_test, predicted))
