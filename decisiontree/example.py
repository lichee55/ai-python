from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
# print(iris.feature_names)
X = iris.data[:, 2:] # petal length and width
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

res=tree_clf.predict_proba([[5, 1.5]])
print(res)
res =tree_clf.predict([[5, 1.5]])
print(res)