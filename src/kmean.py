import numpy as np
import os
from sklearn.cluster import KMeans

def joinPath(a, b):
    return os.path.realpath(os.path.join(a, b))

DIR_NAME = os.path.dirname(__file__)

def loadTrainingData():
    xTrain = []
    with open(joinPath(DIR_NAME, '../data/train_kmean.txt')) as f:
        lines = f.readlines()
        for line in lines:
            xTrain.append([float(v) for v in line.strip().split(' ')])
    return xTrain

xTrain  = loadTrainingData()
xTrain = np.array(xTrain)

model = KMeans(n_clusters=3, random_state=0)
model.fit(xTrain)

# display all cluster center
print(model.cluster_centers_)

print(model.predict(xTrain))

# [[2.  3.5]
#  [1.5 2. ]
#  [3.  3.5]]
# [1 1 0 2 2 0]