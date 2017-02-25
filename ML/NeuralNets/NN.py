from random import random
from random import seed
from math import exp

def initializeNetwork(n_inputs,n_hidden,n_outputs):
    network=list()

    hiddenlayer=[{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    network.append(hiddenlayer)

    outputlayer=[{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(outputlayer)

    return network

seed(1)
'''
network=initializeNetwork(2,1,2)
for i in network:
    print(i)'''


def multiply(weights,inputs):
    result=weights[-1]
    for i in range(len(weights)-1):
        result+=weights[i]*inputs[i]
    return result

def activate(result):
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
'''
row=[1,0,None]
out=feedForward(network,row)
print(out)'''

def sigmoidDerivative(output):
    return output*(1.0-output)

def calculatebackproperror(network,expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors=list()
        if i==len(network)-1:
            for j in range(len(layer)):
                neuron=layer[j]
                error=expected[j]-neuron['output']
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

'''expected=[0,1]
net=[[{'output': 0.7105668883115941,'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
     [{'output': 0.6213859615555266,'weights': [0.2550690257394217, 0.49543508709194095]},
      {'output': 0.6573693455986976,'weights': [0.4494910647887381, 0.651592972722763]}]
     ]
calculatebackproperror(net,expected)
print()
print()
for layer in net:
    print(layer)
'''
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
        for row in data:
            outputs=feedForward(network,row)

            expected=[0.0 for i in range(n_outputs)]
            expected[row[-1]]=1
            sum_error+=sum([(expected[j]-outputs[j])**2 for j in range(len(expected))])

            calculatebackproperror(network,expected)
            updateWeights(network,row,lrate)
        print('>epoch=%d,error=%.3f'%(i,sum_error))


# Make a prediction with a network
def predict(network, row):
    outputs = feedForward(network, row)
    return outputs.index(max(outputs))

print("final")
# Test training backprop algorithm
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initializeNetwork(n_inputs, 2, n_outputs)
trainNetwork(network, dataset, 0.5, 20, n_outputs)
for layer in network:
    print(layer)

for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))

