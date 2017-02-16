'''
Created on 16-Feb-2017

@author: Madhu
'''

class linearClassfier:
    def __init__(self):
        self.X
        self.y
        self.W
        
    
    def SVM(self,x,y,w):
        
        delta=1.0
        
        #calculates w*x for the image--> results in vector of scores for every class
        scores=w.dot(x)
        
        #checking the score for the y(target) value
        predictedScore=scores[y]
        
        # no of classes in weights
        noOfClasses=w.shape(0)
        
        loss=0
        
        for i in range(noOfClasses):
            
            if i==y:
                continue
            loss += max(0, scores[i] - predictedScore + delta)
             
        return loss
    
    def L(self):
        for i in self.X.shape[0]:
            loss=0;
            loss+=self.SVM(self.X[i],self.y[i],self.W)
        return loss    