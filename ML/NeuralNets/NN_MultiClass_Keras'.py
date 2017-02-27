import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split

np.random.seed(7)

#Read the data
dataframe=pd.read_csv('C:/Users/Madhu/Desktop/iris.txt',header=None)
dataset=dataframe.values
X = dataset[:,0:4].astype(float)
Y=dataset[:,4]

#encode string to int(Y label)
encode=LabelEncoder()
encode.fit(Y)
EncodeY=encode.transform(Y)
print(EncodeY)

#Make a vector of boolean values of output
ActualY=np_utils.to_categorical(EncodeY)
print(ActualY)

def createmodel():
    model=Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator=KerasClassifier(build_fn=createmodel,nb_epoch=200,batch_size=5,verbose=0)

'''kfold=KFold(n_splits=10,shuffle=True,random_state=7)

results=cross_val_score(estimator,X,ActualY,cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))'''

X_train, X_test, Y_train, Y_test = train_test_split(X, ActualY, test_size=0.33, random_state=7)

estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)
print(predictions)
print(encode.inverse_transform(predictions))


