from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Flatten

(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

#Reshape the data (Samples,pixels,width,height)
X_train=X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_test=X_test.reshape(X_test.shape[0],1,28,28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#one hot coding for Y values
Y_train=np_utils.to_categorical(Y_train)
Y_test=np_utils.to_categorical(Y_test)
numClasses=Y_train.shape[1]

def createModel():
    model=Sequential()
    model.add(Convolution2D(32,5,5,border_mode='valid',input_shape=(1,28,28),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(15,3,3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(numClasses,activation='softmax'))

    #Compile the model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

#build and fit
model=createModel()
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),nb_epoch=10,batch_size=200,verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

#Save the model

saveJson=model.to_json()
with open('C:/Users/Madhu/Desktop/KerasModel/CNN/LargeCNNDigit.json','w') as file:
    file.write(saveJson)
model.save_weights('C:/Users/Madhu/Desktop/KerasModel/CNN/LargeCNNDigit.h5')
