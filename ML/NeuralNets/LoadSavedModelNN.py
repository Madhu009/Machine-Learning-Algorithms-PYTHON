from keras.models import model_from_json
import numpy as np

dataset=np.loadtxt("C:/Users/Madhu/Desktop/pima.txt",delimiter=',')
X=dataset[:,0:8]
Y=dataset[:,8]

loadJson=open('C:/Users/Madhu/Desktop/pima.json','r')
modelFromJson=loadJson.read()
loadJson.close()

loadedModel=model_from_json(modelFromJson)
loadedModel.load_weights('C:/Users/Madhu/Desktop/pima.h5')

loadedModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
scores1=loadedModel.evaluate(X,Y)
print("%s:%.2f%%" % (loadedModel.metrics_names[1],scores1[1]*100))

#Predict the input
predictions=loadedModel.predict(X)
print(predictions)

# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)