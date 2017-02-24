
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse

def sigmoid_activation(x):
	return 1.0 / (1 + np.exp(-x))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 250 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=400, n_features=2, centers=2,
	cluster_std=2.5, random_state=95)


X = np.c_[np.ones((X.shape[0])), X]

print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1],))

lossHistory = []

def nextBatch(x,y,size):
    for i in np.arange(0,x.shape[0],size):
        yield (x[i:i+size],y[i:i+size])

for epoch in np.arange(0, args["epochs"]):
    epochloss=[]
    for (batchX,batchY) in nextBatch(X,y,size=5):

        preds = sigmoid_activation(batchX.dot(W))

        error = preds -batchY

        loss = np.sum(error ** 2)
        epochloss.append(loss)

        gradient = batchX.T.dot(error) /batchX.shape[0]
        W += -args["alpha"] * gradient
    lossHistory.append(np.average(epochloss))
    #print("[INFO] epoch #{}, loss={:.7f}".format(epoch + 1, loss))

for i in np.random.choice(400, 10):

	activation = sigmoid_activation(X[i].dot(W))

	label = 0 if activation < 0.5 else 1

	# show our output classification
	print("activation={:.4f}; predicted_label={}, true_label={}".format(
		activation, label, y[i]))

Y = (-W[0] - (W[1] * X)) / W[2]

# plot the original data along with our line of best fit
plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
plt.plot(X, Y, "r-")

# construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()