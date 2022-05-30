# from matplotlib.pyplot import yticks
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def joinPath(a, b):
    return os.path.realpath(os.path.join(a, b))

DIR_NAME = os.path.dirname(__file__)

def convertStringToList(data:str, dauPhanCach=' ', convertFunc=float):
    if convertFunc == None:
        return data.split(dauPhanCach)
    return [convertFunc(v) for v in data.split(dauPhanCach)]

def loadTrainingData():
    xTrain = []
    yTrain = []
    xTest = []
    yTest = []

    with open(joinPath(DIR_NAME, '../data/train.txt')) as f:
        lines = f.readlines()
        # print(lines[0].replace('\n', '').split(' '))

        xTrain.append(convertStringToList(lines[0].replace('\n', '')))
        yTrain.append(convertStringToList(lines[1].replace('\n', ''), convertFunc=None))
        xTest.append(convertStringToList(lines[2].replace('\n', '')))
        yTest.append(convertStringToList(lines[3].replace('\n', ''), convertFunc=None))
    
    return xTrain, yTrain, xTest, yTest

xTrain, yTrain, xTest, yTest = loadTrainingData()

xTrain = np.array(xTrain).T
xTest = np.array(xTest).T


lbTransformer = LabelEncoder()
lbTransformer.fit(np.array(yTrain).flatten())

yTrainShape = np.array(yTrain).shape
yTrain = lbTransformer.transform(np.array(yTrain).flatten())
yTrain = np.array(yTrain).reshape(yTrainShape)
yTrain = np.array(yTrain).T

yTestShape = np.array(yTest).shape
yTest = lbTransformer.transform(np.array(yTest).flatten())
yTest = np.array(yTest).reshape(yTestShape)
yTest = np.array(yTest).T

model = LogisticRegression()
model.fit(xTrain, yTrain)

print(model.classes_)
print(model.coef_)
result = model.predict(xTest)

print(mean_squared_error(yTest, result))
print(lbTransformer.inverse_transform(result.flatten()))