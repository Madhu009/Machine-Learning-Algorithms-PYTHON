import code,numpy as np

def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoidderivate(X):
    return X*(1-X)

