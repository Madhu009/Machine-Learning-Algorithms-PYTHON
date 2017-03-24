import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from numpy import array

from random import random
from random import seed
from math import exp

def fun(filename):
    data = pd.read_csv(filename)
    data = data.fillna(0)
    feature_cols = ['loan_amnt', 'funded_amnt', 'int_rate', 'annual_inc', 'dti', 'open_acc', 'revol_bal', 'revol_util', 'total_acc',
                    'total_rec_int', 'tot_cur_bal', 'total_rev_hi_lim','loan_status']

    X = data[feature_cols]
    # Feature scale X=X/dys*hurs 365*24
    X['annual_inc'] = X['annual_inc'] / 8760

    # Diffrence of bank and investor
    X['funded_amnt'] = X['funded_amnt'] - data['funded_amnt_inv']

    # Loan amount / term
    X['loan_amnt'] = X['loan_amnt'] / 1000

    X['tot_cur_bal'] = X['tot_cur_bal'] / 100000
    X['total_rev_hi_lim'] = X['total_rev_hi_lim'] / 100000
    X['total_rec_int'] = X['total_rec_int'] / 1000
    X['revol_bal'] = X['revol_bal'] / 1000
    X['revol_util'] = X['revol_util'] / 1000
    X['total_acc'] = X['total_acc'] / 10
    X['dti'] = X['dti'] / 10
    X=X.values
    X=np.array(X.tolist())
    return X

file1="C:/Users/Madhu/Desktop/train_indessa.csv"
dataset=fun(file1)





#Neural net
def initializeNetwork(n_inputs,n_hidden,n_outputs):
    network=list()

    hiddenlayer=[{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    network.append(hiddenlayer)

    outputlayer=[{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(outputlayer)

    return network

seed(1)

def multiply(weights,inputs):
    result=weights[-1]
    for i in range(len(weights)-1):
        result+=weights[i]*inputs[i]
    return result

def activate(result):
    if result<0:
        return 1-1/(1+exp(result))
    return 1.0/(1.0+exp(-result))

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


def sigmoidDerivative(output):
    return output*(1.0-output)

def calculatebackproperror(network,expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors=list()
        if i==len(network)-1:
            for j in range(len(layer)):
                neuron=layer[j]
                error=expected-neuron['output']
                errors.append(error)
        else:
            for j in range(len(layer)):
                herror=0
                for neuron in network[i+1]:
                    herror+=(neuron['weights'][j]*neuron['delta'])
                errors.append(herror)
        for j in range(len(layer)):
            neuron=layer[j]
            neuron['delta']=errors[j]*sigmoidDerivative(neuron['output'])


def updateWeights(network,input,lrate):
    for i in range(len(network)):
        inputs = input[:-1]
        if i!=0:
            inputs=[neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]
            neuron['weights'][-1] += lrate * neuron['delta']


def trainNetwork(network,data,lrate,epoch,n_outputs):

    for i in range(epoch):
        sum_error=0
        c=0
        for row in data:
            c+=1
            if(c%15000==0):
                print(c)
            outputs=feedForward(network,row)
            expected=row[-1]
            sum_error+=(expected-outputs)**2

            calculatebackproperror(network,expected)
            updateWeights(network,row,lrate)
            print(sum_error)
        print('>epoch=%d,error=%.3f'%(i,sum_error))


# Make a prediction with a network
def predict(network, row):
    outputs = feedForward(network, row)
    return outputs

print("final")
# Test training backprop algorithm
seed(1)

n_inputs=len(dataset[0])-1
network=initializeNetwork(n_inputs,16,1)
trainNetwork(network, dataset, 5, 10, 1)

for layer in network:
    print(layer)

for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction[0]))

