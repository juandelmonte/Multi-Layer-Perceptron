# %%
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pandas as pd
from matplotlib import pylab
from main.MultiLayerPerceptron import MLP
from main.utils import *
from helpers.mnist import *


##DataSet -> https://drive.google.com/drive/folders/121IvgXaUqL3iQ5v5EJWmcaHK7gQP8AaJ?usp=sharing

#train
filename = os.path.join(currentdir, 'data\\mnist_train.csv')
df = pd.read_csv(filename,header=None)
input_train,output_train = encode(df)
#test
filename = os.path.join(currentdir, 'data\\mnist_test.csv')
df = pd.read_csv(filename,header=None)
input_test,output_test = encode(df)


model = MLP([len(input_train[0]),170,10], weights_multiplier = 0.01)

#initial test acc
model.setData(input_test,output_test)
print('initial test cost:', model.getCost(), 'initial test accuracy:', getAccuracy(model))

#train
model.setData(input_train,output_train)
model.learning_rate=0.2

print('initial training cost:', model.getCost(), ' initial training accuracy:', getAccuracy(model))

model.lastLayer.activation = logistic
model.cost = cross_entropy

model.train(2000,200,20)

#test data

model.setData(input_test,output_test)
print('test cost:', model.getCost(), ' test accuracy:', getAccuracy(model))