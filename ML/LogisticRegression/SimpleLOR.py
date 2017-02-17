from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris=datasets.load_iris()

x,y=iris.data[:-1:],iris.target[:-1]

logist=LogisticRegression()
logist.fit(x,y)

print(logist.predict(iris.data[-1,:]).reshape(x.shape))
