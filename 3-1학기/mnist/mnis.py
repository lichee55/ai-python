from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes  import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


digits = fetch_openml('mnist_784', version=1, cache=True)

# from sklearn.datasets import load_digits
# digits = load_digits()

X, y = digits["data"], digits["target"]
print(X.shape) # (70000, 784)
print(y.shape) # (70000,)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 30)

# 모델 학습
gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = gnb_clf.predict(X_test)

# 정확도 계산
accuracy = 100.0 * (y_test==y_pred).sum() / X_test.shape[0]
print("Accuracy : ", round(accuracy, 2), "%") 

# performance measures
# accuracy = cross_val_score(gnb_clf, X, y, cv=3, scoring="accuracy")
# print(accuracy) # [0.55858404 0.56267947 0.54720782]

# confusion matrix
y_pred = cross_val_predict(gnb_clf, X, y, cv=3)
confusion_matrix = confusion_matrix(y, y_pred)
# print(confusion_matrix)

# visualize confusion matrix
# from matplotlib import pyplot as plt
# import numpy as np

# plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.gray)
# plt.title('Confusion matrix')
# plt.colorbar()
# ticks = np.arange(5)
# plt.xticks(ticks, ticks)
# plt.yticks(ticks, ticks)
# plt.ylabel('True labels')
# plt.xlabel('Predicted labels')
# plt.show()

from sklearn.model_selection import cross_val_score

accuracy = cross_val_score(gnb_clf, X, y, cv=3, scoring="accuracy")
precision = cross_val_score(gnb_clf, X, y, cv=3, scoring="precision_weighted")
recall = cross_val_score(gnb_clf, X, y, cv=3, scoring="recall_weighted")
f1 = cross_val_score(gnb_clf, X, y, cv=3, scoring="f1_weighted")

from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(gnb_clf, X, y, cv=3)
print(classification_report(y, y_pred))
