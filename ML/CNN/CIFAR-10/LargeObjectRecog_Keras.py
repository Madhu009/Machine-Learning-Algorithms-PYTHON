from keras.datasets import cifar10
import matplotlib.pyplot as plt
from scipy.misc import toimage
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from keras.constraints import maxnorm
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np

np.random.seed(7)

(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
'''
for i in range(0,9):
    plt.subplot(3,3,1+i)
    plt.imshow(toimage(X_train[i]))
plt.show()'''

#Reshape the data
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#one hot coding for Y values
Y_train=np_utils.to_categorical(Y_train)
Y_test=np_utils.to_categorical(Y_test)
numClasses=Y_train.shape[1]

#Model starts from here
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(numClasses, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=epochs, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#Save the model

jsonFile=model.to_json()
with open('C:/Users/Madhu/Desktop/KerasModel/CNN/CIFAR10/Largeobject.json','w') as file:
    file.write(jsonFile)
model.save_weights('C:/Users/Madhu/Desktop/KerasModel/CNN/CIFAR10/Largeobject.h5')