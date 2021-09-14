from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 학습 데이터 로딩
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

# 원 데이터를 이용하여 학습 및 성능 측정
rnd_clf.fit(X_train, y_train)
y_test_pred = rnd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print("축소 전 정확도 =", round(accuracy, 2))

# 차원 축소
lda = LinearDiscriminantAnalysis(n_components = 2)
lda.fit(X_train, y_train)
X_train_reduced = lda.transform(X_train)
X_test_reduced = lda.transform(X_test)

# 축소 데이터 이용하여 학습 및 성능 측정
rnd_clf.fit(X_train_reduced, y_train)
y_test_pred_reduced = rnd_clf.predict(X_test_reduced)
accuracy = accuracy_score(y_test, y_test_pred_reduced)
print("축소 후 정확도 =", round(accuracy, 2))
