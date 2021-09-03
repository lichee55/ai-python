from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_moons
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score  # 교차 검증을 위한 모듈
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# 데이터 셋 로드
my_data = pd.read_csv("C:/gitRepo/ai-python/last/credit-g.csv")

cut_data = pd.concat(
    [my_data['duration'], my_data['age'], my_data['class']], axis=1)

undivide_input = pd.concat([my_data['duration'], my_data['age']], axis=1)
undivide_label = cut_data['class']

# 데이터 train set과 test set 분리
train_set, test_set = train_test_split(
    cut_data, train_size=0.8, random_state=30)

# SVM
svm_clf = Pipeline((("scaler", StandardScaler()),
                   ("linear_svc", LinearSVC(C=1, loss="hinge")),))
# svm_clf.fit(train_set_inputs, train_set_label)

accuracy = cross_val_score(svm_clf, undivide_input,
                           undivide_label, cv=3, scoring="accuracy")
print(accuracy)