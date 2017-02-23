
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
    return x,y

#find hypothsis(prediction value)
def predict(row,weights):
    output=weights[0]
    for i in range(len(row)):
        output+=weights[i+1]*row[i]
    return output





filename='C:/Users/Madhu/Desktop/Book1.csv'
data=loadCSV(filename)
convertToFloat(data)
print(data)
x,y=getData(data)
print(x)