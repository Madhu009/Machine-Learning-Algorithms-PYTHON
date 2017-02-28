# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from keras.models import model_from_json

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
 

#Load data
dataset=np.loadtxt("C:/Users/Madhu/Desktop/pima.txt",delimiter=',')
X=dataset[:,0:8]
Y=dataset[:,8]

#Define model
model=Sequential()
model.add(Dense(12,input_dim=8,init='uniform',activation='relu'))
model.add(Dense(8,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(X,Y,nb_epoch=150,batch_size=10)

#Evaluate the model
scores=model.evaluate(X,Y)
print("%s:%.2f%%" % (model.metrics_names[1],scores[1]*100))

#Predict the input
predictions=model.predict(X)
print(predictions)

# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

#Save the model
modelJSON=model.to_json()
with open('C:/Users/Madhu/Desktop/pima.json','w') as jsonfile:
    jsonfile.write(modelJSON)

model.save_weights('C:/Users/Madhu/Desktop/pima.h5')

