from keras.models import model_from_json
from keras.datasets import mnist
from keras.utils import np_utils

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Load data
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

plt.imshow(X_test[4],cmap=plt.get_cmap('gray'))
plt.show()

#Reshape X cause input is image
totalPixels=X_train.shape[1]*X_train.shape[2]
X_train=X_train.reshape(X_train.shape[0],totalPixels).astype('float32')
X_test=X_test.reshape(X_test.shape[0],totalPixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#encode string to int(Y label)
encode=LabelEncoder()
encode.fit(Y_test)
EncodeY=encode.transform(Y_test)
print(EncodeY)

#multiclass so convert it to vector of output
Y_train=np_utils.to_categorical(Y_train)
Y_test=np_utils.to_categorical(Y_test)
num_classes=Y_train.shape[1]



file=open('C:/Users/Madhu/Desktop/KerasModel/Digit.json','r')
loadedModel=file.read()
file.close()

model=model_from_json(loadedModel)
model.load_weights('C:/Users/Madhu/Desktop/KerasModel/Digit.h5')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
scores=model.evaluate(X_test,Y_test,verbose=0)
print("%s:%.2f%%" % (model.metrics_names[1],(100-scores[1]*100)))


pred=model.predict(X_test)

for p in pred:
    listp = p.tolist()
    print(listp.index(max(listp)))