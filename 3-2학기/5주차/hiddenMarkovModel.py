import numpy as np
from hmmlearn import hmm

X = [[1],[1],[1],[2],[2],[2],[2],[1],[1],[1],[2],[2],[2],[2],[1],[1],[1],[2],[2],[2],[2]]
model = hmm.MultinomialHMM(2)
model.fit(X)

Y1 = [[1],[1],[1],[1]]
Y2 = [[1],[1],[1],[2]]

print(model.score(Y1))
print(model.score(Y2))