import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities import visualize_classifier

# Load input data
input_file = 'data_imbalance.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Split data into training and testing datasets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Extremely Random Forests classifier
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0, 'class_weight': 'balanced'}
classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, ytest, '테스트 셋')

# Evaluate classifier performance
class_names = ['클래스0', '클래스1']
print("#"*50)
print("\n테스트 셋에 대한 분류 성능\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*50 + "\n")
