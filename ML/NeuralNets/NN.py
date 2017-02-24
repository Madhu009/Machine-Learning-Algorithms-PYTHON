from random import random
from random import seed
import numpy as np
def initializeNetwork(n_inputs,n_hidden,n_outputs):
    network=list()

    hiddenlayer=[{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    network.append(hiddenlayer)

    outputlayer=[{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(outputlayer)

    return network

seed(1)
network=initializeNetwork(2,1,2)
for i in network:
    print(i)


def multiply(weights,inputs):
    result=weights[-1]
    for i in range(len(weights)-1):
        result+=weights[i]*inputs[i]
    return result

def activate(result):
    return 1.0/(1.0+np.exp(-result))

def feedForward(network,row):
    input = row
    for layer in network:
        newInput = []
        for neuron in layer:
            value=multiply(neuron['weights'],input)
            neuron['output']=activate(value)
            newInput.append(neuron['output'])
        input=newInput
    return input

row=[1,0,None]
out=feedForward(network,row)
print(out)