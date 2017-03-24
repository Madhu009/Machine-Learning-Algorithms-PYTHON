from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from numpy import array
iris=datasets.load_iris()

x,y=iris.data[:-1:],iris.target[:-1]

logist=LogisticRegression()
print(type(x))
logist.fit(x,y)
j=iris.data[-1,:]
print(j)
print(logist.predict(j))

