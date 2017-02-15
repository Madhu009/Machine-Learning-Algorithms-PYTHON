'''
Created on 15-Feb-2017

@author: Madhu
'''
import numpy as np
from numpy import dtype

class NNC:
    def __init__(self):
        pass
    
    def train(self,X,Y):
        
        #variables accessed globally
        self.XValue=X
        self.YValue=Y
    
    def predict(self,X):
        
        noOfTests=X.shape[0] # returns dimensions if X is a matrix of n*m then shape[0]=n
        
        #check to see if output type matches the input type
        
        YPredicted=np.zeros(noOfTests,dtype=self.YValue.dtype) # gives the same data type with no of zeors
        
        #Loop all the test X's to calculate the L1 distance from train images
        
        for i in range(noOfTests):
            
            # L1 distance (ith test image and XValue train image)
            distances=np.sum(np.abs(self.XValue-X[i:]), axis=1)
            
            #get the min distance
            min_index=np.argmin(distances)
            
            YPredicted[i]=self.YValue[min_index]
            
            return YPredicted
    