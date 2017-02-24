from sklearn.datasets.samples_generator import make_moons
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x,y=make_moons(200,noise=0.20)
plt.scatter(x[:,0],x[:,1],s=40,c=y,cmap=plt.cm.spec)
