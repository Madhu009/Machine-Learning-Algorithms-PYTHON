import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

np.random.seed(7)

#Load data
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

#Reshape X cause input is image
totalPixels=X_train.shape[1]*X_train.shape[2]
X_train=X_train.reshape(X_train.shape[0],totalPixels).astype('float32')
X_test=X_test.reshape(X_test.shape[0],totalPixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#multiclass so convert it to vector of output
Y_train=np_utils.to_categorical(Y_train)
Y_test=np_utils.to_categorical(Y_test)
num_classes=Y_train.shape[1]


def createModel():
    model=Sequential()
    model.add(Dense(totalPixels,input_dim=totalPixels,init='normal',activation='relu'))
    model.add(Dense(num_classes,init='normal',activation='softmax'))

    #Compile model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model=createModel()
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),nb_epoch=10,batch_size=20)

scores=model.evaluate(X_test,Y_test,verbose=0)
print("%s:%.2f%%" % (model.metrics_names[1],(100-scores[1]*100)))

#Save the model
saveModel=model.to_json()
with open('C:/Users/Madhu/Desktop/KerasModel/Digit.json','w') as jsonfile:
    jsonfile.write(saveModel)
model.save_weights('C:/Users/Madhu/Desktop/KerasModel/Digit.h5')
