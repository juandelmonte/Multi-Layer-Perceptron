import random
import numpy as np
from main.utils import *
import json
import os
dirname = os.path.dirname(__file__)

class layer(object):
        def __init__(self,b=np.array([]),W=np.matrix([])):
            self.W = W #Matrix
            self.b = b #Array

            #list of Arrays (one per training sample)
            self.input = None
            self.z = None
            self.a = None
            self.dC_dz = None
            #self.a_corrected = None # first used for propagating error correction, the error of the previous layer is the difference between its output and the value of this corrected (propagated through its gradient) value. Different layers could have different cost functions

            #activation
            self.activation = np.vectorize(relu)
            self.activationName = 'relu' # only for saving, np.vectorize can't be serialized with __name__.. better maybe create a class activation

            self.n_nodes = len(self.b)

#Numpy MLP
class MLP(object):
    def __init__(self, structure, name = 'MLP', weights_multiplier = 0.1):
        self.name = name

        self.layers = []
        self.input = None
        self.output = None
        self.n_training = None
        self.cost = np.vectorize(MSE)

        self.structure = structure
        self.fromPandas = False

        #initialize layers with bias equal to 0 and random weights [shape (amountOfNodes,amountOfNodesPrev)] - first value is not layer but number of inputs
        for a in range(1,len(structure)):
            layer_n = structure[a]
            n_inputs = structure[a-1]

            W = np.matrix([[random.random() * weights_multiplier for _ in range(n_inputs)] for __ in range((layer_n))])
            self.layers.append(layer(np.array([0 for _ in range(layer_n)]),W))

        self.lastLayer = self.layers[-1]
        self.firstLayer = self.layers[0]
        self.n_layers = len(structure)-1

        self.learning_rate = 0.1
    
    def setInput(self, input):
        if self.fromPandas: #input is a DataFrame
            self.input = input.to_numpy()
        else:
            self.input = [np.array(i) for i in input] #I will pay this price for having everything being a numpy array and operate smoothly, better would be if the input is already all formated beforehand --> pd.to_numpy() to the main file and the sample that one, would save time

        self.firstLayer.input = self.input
        self.n_training = len(input)
    
    def setOutput(self, output):
        if self.fromPandas:
            self.output = output.to_numpy()
        else:
            self.output = [np.array(o) for o in output] #Same as input

    def setData(self, input, output):
        self.setInput(input)
        self.setOutput(output)
  
    def feedforward(self):
        for layer_inx in range(self.n_layers):
            layer = self.layers[layer_inx]
            
            layer.z = [np.add(layer.W.dot(input),layer.b).A1 for input in layer.input] 
            layer.a = [layer.activation(z_n) for z_n in layer.z]
            
            if not layer_inx == self.n_layers-1:
                self.layers[layer_inx+1].input = layer.a

    def backwardprop(self):
        #calculate neural network output 'error' -> 'error' = dC/dz = dC/da * da/dz
        self.lastLayer.dC_dz=[]
        for training_inx in range(self.n_training):
            da_dz = self.lastLayer.activation(self.lastLayer.z[training_inx], derivative = True)
            dC_da = self.cost(self.lastLayer.a[training_inx], self.output[training_inx], derivative = True)

            self.lastLayer.dC_dz.append(np.array(np.multiply(da_dz,dC_da)))

        #propagate errors to previous layers
        for layer_inx in reversed(range(self.n_layers-1)):
            layer = self.layers[layer_inx]
            layer_next = self.layers[layer_inx+1]
            
            #dC_dz[prev] =  W.T.dot(dC_dz) * da[prev]/dz[prev]
            layer.dC_dz = []
            for training_inx in range(self.n_training):
                propagated = layer_next.W.T.dot(layer_next.dC_dz[training_inx])
                da_dz = layer.activation(layer.z[training_inx], derivative = True)

                layer.dC_dz.append(np.multiply(propagated, da_dz).A1)

            
        #update W y b
        for layer in reversed(self.layers):

            #update weights                      

            R = [np.matrix([np.full(layer.n_nodes,1)]).T.dot(np.matrix(layer.input[training_inx])) for training_inx in range(self.n_training)]

            full=np.full(layer.W.T.shape,1)

            W_temp = np.matrix(np.zeros(layer.W.shape))
            for training_inx in range(self.n_training):
                W_temp = np.add(W_temp,np.multiply(np.multiply(full,layer.dC_dz[training_inx]).T,R[training_inx]))
            
            W_temp = -1 * self.learning_rate * W_temp / self.n_training            
            # in case of L regularization this could be a good place

            layer.W = np.add(layer.W,W_temp)

            #update biases

            gradients_b = layer.dC_dz

            b_temp = np.zeros(layer.n_nodes)
            for b in gradients_b:
                b_temp = np.add(b_temp,b)
            
            b_temp = -1 * self.learning_rate * b_temp / self.n_training

            layer.b = np.add(layer.b,b_temp)         

    def getCost(self):
        self.feedforward()
        layer = self.lastLayer
        cost = np.matrix([self.cost(layer.a[training_inx],self.output[training_inx]) for training_inx in range(self.n_training)])        
        return cost.mean()

    def print(self, bool = True): # similar could be used to save previous state
        layers=[]
        for inx,layer in enumerate(self.layers):
            layers.append({'Layer Num' : inx, 'W': layer.W.tolist(), 'b': layer.b.tolist(), 'Activation:': layer.activationName, 'W mean and std': (layer.W.mean(),layer.W.std()), 'b mean and std':(layer.b.mean(),layer.b.std())})
        jsonStr = json.dumps(layers, indent=3)
        if bool:
            print(jsonStr)
        else:
            return layers

    def save(self, id):
        filename = os.path.join(dirname, 'output/' + self.name + '_' + str(id) + '.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.print(False), f, ensure_ascii=False, indent=4)

        return True

    def load(orig):
        pass

    def train(self, iterations = 10000, evaluation_loop = 100, n_samples = 10, replacement = True, learning_rate_schedule = None):
        #store initial values
        IN = self.input.copy()
        OUT = self.output.copy()

        cost_init = self.getCost()
        for i in range(iterations): 
            for sampled_input, sampled_output in sample(IN,OUT,n_samples, replacement): # in case of replacement = False (minibatch) iterations is epocs
                self.setData(sampled_input,sampled_output)
                cost_init_loop = self.getCost()  #getCost is applying feedforward
                self.backwardprop()
                if i%evaluation_loop==0: # I actually don't want to do this every step eg.-> if i%100==0
                    cost_end_loop = self.getCost()

                    cost_ratio = cost_init_loop/cost_end_loop
                    print(i, cost_init_loop, cost_end_loop, cost_ratio, self.learning_rate)

                    if cost_ratio<1:
                        self.learning_rate = self.learning_rate*0.8

                    #there could be a procedure to monitor generalization error here

        self.setData(IN, OUT)
        cost_end = self.getCost()

        print('initial and final training cost: ', cost_init, cost_end)
        return True
            
    def getAccuracy(self): # only for classification
        pass