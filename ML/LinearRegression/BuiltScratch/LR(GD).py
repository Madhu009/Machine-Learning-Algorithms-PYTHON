import numpy as np
import matplotlib.pyplot as plt
from csv import reader
#Load a file and store the data
def loadCSV(filename):
    data=list()
    with open(filename,'r') as file:
        csvReader=reader(file)
        for row in csvReader:
            if not row:
                continue
            data.append(row)
    return data

#convert the string values into float
def convertToFloat(data):
    for row in range(len(data)):
        for col in range(len(data[0])):
            data[row][col]=float(data[row][col])
    return data

#get features and targets
def getData(data):
    x=list()
    y=list()
    for row in range(len(data)):
        example=list()
        for col in range(len(data[0])):
            if col is len(data[0])-1:
                y.append(data[row][col])
            else:
                example.append(data[row][col])
        x.append(example)
    x=np.c_[np.ones(len(x)),x]
    y=np.asarray(y)
    return x,y

#find hypothsis(prediction value)
def predict(x,weights):
    return x.dot(weights)

#loop over all examples and find error
def GD(x,y,lrate,epochs):
    W=np.random.uniform(size=x.shape[1],)
    for i in range(epochs):

        predicted=predict(x,W)

        error=predicted-y

        totalerror=np.sum(error**2)

        gradient=x.T.dot(error)/x.shape[0]
        print(totalerror)
        W+=-lrate*gradient
    return W

filename='C:/Users/Madhu/Desktop/Book1.csv'
data=loadCSV(filename)
convertToFloat(data)
x,y=getData(data)
print(x,y)

weight=GD(x,y,0.001,50000)
print(weight)
input=[10,5]
out=weight[0]
for i in range(len(input)):
    out+=weight[i+1]*input[i]

print(out)
