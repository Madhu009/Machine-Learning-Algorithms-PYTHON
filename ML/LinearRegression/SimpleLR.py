# this program predicts the hosing price

from sklearn import linear_model
import matplotlib.pyplot as plot
import pandas as pd

#fuction to read the data from file
def readData(filename):
    data=pd.read_csv(filename,encoding = "ISO-8859-1",sep='\s*,\s*',header=0, engine='python')
    x=[]
    y=[]
    for x_data, y_data in zip(data['square_feet'], data['price']):
        x.append([float(x_data)])
        y.append([float(y_data)])
    return x,y

#fuction to plot the data

def showModel(x,y1):
    lr=linear_model.LinearRegression()
    lr.fit(x, y1)
    #ploting the points
    plot.scatter(x,y,color='blue')
    #drawing a line
    plot.plot(x,lr.predict(x),color='red',linewidth=4)
    plot.xticks()
    plot.yticks()
    plot.show()

def model(x,y,input):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    predict_outcome = regr.predict(input)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions

x,y=readData("C:/Users/Madhu/Desktop/test.csv")
print(x)
print(y)
print(model(x,y,160))
showModel(x,y)

