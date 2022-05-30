import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# # Các phép đo phổ biến cho bài toán regression
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# # Các phép đo phổ biến cho bài toán classification
# from sklearn.metrics import accuracy_score, f1_score

# Ridge(alpha=0.1, normalize=True).fit(x_train, y_train)
# Lasso(alpha=0.001, normalize=True).fit(x_train, y_train)

def makeXBar(x):
    n = len(x)
    oneArr = np.ones((n, 1))
    return np.concatenate((oneArr, x), axis=1)

# print(type(makeXBar([[1, 2, 5],[3, 4, 6]])))

def getData(n:int, inputFunc=input, convertFunc=float):
    data = []
    for i in range(n):
        data.append([convertFunc(v) for v in inputFunc().split(' ')])
    return data

def splitData(data, n:int):
    data = np.array(data)
    x = data[:, :n]
    y = data[:, n:]
    return x,y

def main():
    #train
    lengthX, n = getData(1, input, int)[0]
    data = getData(n)
    x_train, y_train = splitData(data, lengthX)
    
    #test
    nTestX = int(input())
    testX = getData(nTestX)
    testX = np.array(testX)
    
    #predict
    linearModel = LinearRegression().fit(x_train, y_train)
    result = linearModel.predict(testX)
    result = np.array(result).flatten()

    for v in result:
        print(round(v, 2))

main()
