import matplotlib.pyplot as plt
import numpy as np
import random

def logistic(x: np.array, derivative = False):
    if not derivative:
        return 1/(1+np.exp(-x))
    else:
        return logistic(x)*(1-logistic(x))

def relu(x, derivative = False):
    if not derivative:
        if x>0:
            return x
        else:
            return 0.2*x
    else:
        if x>0:
            return 1
        else:
            return 0.2

def linear(x, derivative = False):
    if not derivative:
        return x
    else:
        return 1
        
def MSE(x,y, derivative = False):
    if not derivative:
	    return (x-y)**2
    else:
	    return 2*(x-y)

def cross_entropy(a, y, derivative = False):
    if not derivative:
        return -(y*np.log(a)+(1-y)*np.log(1-a))
    else:
        return -1*((a-y)/((a-1)*a))

def normalize_linear(array:list):
	max = np.max(array)
	min = np.min(array)
	diff = max-min

	result = []
	for a in array:
		result.append((a-min)/diff)
	return result

def sample(a, b, N=10, replacement = True): # not replacement : minibatch
    if replacement:
        c=random.sample(range(len(a)),N)

        anew=[]
        bnew=[]

        for e in c:
            
            anew.append(a[e])
            bnew.append(b[e])
        
        yield anew,bnew

def rSquared(observed, predicted):
    observed = np.array(observed)
    predicted = np.array(predicted)
    error = ((predicted - observed)**2).sum()
    meanError = error/len(observed)
    return 1 - (meanError/np.var(observed))

class iterator(object): #to be used in the learning rate schedule
    def __init__(self, lst):
        self.lst = lst
        self.end = len(self.lst)-1

        self.inx = 0
        self.value = lst[self.inx]

        return self.value
    
    def next(self):
        self.inx += 1
        self.value = self.lst[self.inx]

        if self.inx == self.end:
            self.inx = -1        
        
        return self.value
