import pandas as pd
import numpy as np

def readArrays():
    # Create an array of ones
    print(np.ones((3, 4)))

    # Create an array of zeros
    print(np.zeros((2, 3, 4), dtype=np.int16))

    # Create an array with random values
    print(np.random.random((2, 2)))

    # Create an empty array
    print(np.empty((3, 2)))

    # Create a full array
    print(np.full((2, 2), 7))

    # Create an array of evenly-spaced values
    print(np.arange(10, 25, 5))

    # Create an array of evenly-spaced values
    np.linspace(0, 2, 9)






def readData(filename):
    data = pd.read_csv(filename, encoding="ISO-8859-1", sep='\s*,\s*', header=0, engine='python')
    x=[]
    y=[]
    for feature,target in zip(data['size'],data['price']):
        x.append(feature)
        y.append(target)

    return x,y


def readDataNP(filename):
    x,y=np.loadtxt(filename,skiprows=1,unpack=True)
    print(x)




filename='C:/Users/Madhu/Desktop/Book1.csv'
x,y=readData(filename)
#print(x)
#readArrays()
readDataNP('C:/Users/Madhu/Desktop/b.txt')