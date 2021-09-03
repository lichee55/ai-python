import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

# Load input data
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Split the data into training and testing datasets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Define the parameter grid 
parameter_grid = [ {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]}, {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]} ]
metrics = ['precision_weighted', 'recall_weighted']
for metric in metrics:
    print("\n***", metric,"에 대한 최적의 패러미터 찾기 ***")

    classifier = grid_search.GridSearchCV(ExtraTreesClassifier(random_state=0), 
            parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    print("\n 패러미터 조합에 대한 성능 : ")
    for params, avg_score, _ in classifier.grid_scores_:
        print(params, '-->', round(avg_score, 3))

    print("\n최적의 패러미터 : ", classifier.best_params_)

    y_pred = classifier.predict(X_test)
    print("\n 테스트 데이터에 대한 성능 : \n")
    print(classification_report(y_test, y_pred))
