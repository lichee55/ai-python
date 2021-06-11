import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prp  # class Lable encoder
from sklearn.naive_bayes import GaussianNB  # Naive-Bayes : Line18~
from sklearn.model_selection import cross_val_score  # 교차 검증을 위한 모듈

# 데이터 셋 로드
my_data = pd.read_csv("C:/gitRepo/ai-python/last/credit-g.csv")

cut_data = pd.concat(
    [my_data['duration'], my_data['age'], my_data['class']], axis=1)

undivide_input = pd.concat([my_data['duration'], my_data['age']], axis=1)
undivide_label = cut_data['class']

# 데이터 train set과 test set 분리
train_set, test_set = train_test_split(
    cut_data, train_size=0.8, random_state=30)

# Naive-Bayes
gnb_clf = GaussianNB()

# Naive-Bayes 정확도 계산
accuracy = cross_val_score(gnb_clf, undivide_input,
                           undivide_label, cv=3, scoring="accuracy")
print("Accuracy : ", accuracy)