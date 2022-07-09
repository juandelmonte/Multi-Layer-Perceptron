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
from helpers.YatchDesign import *

# %%
df_total = pd.read_csv(currentdir+'\\data\\yacht_hydrodynamics.csv',sep=';')
df_total

# %%
#leave 40% of the data for testing
df = df_total.sample(frac = 0.60)

#normalize
df_means = df.mean()
df_stds = df.std()

df=(df-df_means)/(df_stds) 

#set training and validation data
input, output = encode(df,'resistance')
input_training, output_training, input_validation, output_validation = generate_samples(input, output,4)

# %%
model = MLP([6,4,1])
model.save('init')

model.setData(input_validation,output_validation)
print('cost of validation data: ',model.getCost())

# %%
model.setData(input_training,output_training)
model.train(10000,1000,15)

model.setData(input_validation,output_validation)
print('cost of validation data: ',model.getCost())

# %%
##test

df_test = pd.DataFrame(df_total.drop(df.index))
output_real=df_test['resistance'].tolist()

# normalization (uses mean and std from training and validation, which is the one the model is trained with)
# aka it's normalized using the same parameters the model used to train:
df_test=(df_test-df_means)/(df_stds)
input_test, output_test = encode(df_test,'resistance')
model.setData(input_test,output_test)
print('test cost:',model.getCost())

# %%
#denormalize  --> inverse df=(df-df_means)/(df_stds) for resistance
mean_resistance = df_means['resistance']
std_resistance = df_stds['resistance']

output_predicted=[a[0]*std_resistance+mean_resistance for a in model.layers[-1].a]

#model.save('')

r2=rSquared(output_real,output_predicted)
pylab.plot(output_real,output_predicted, 'bo')
a =[a for a in range(65)]
pylab.plot(a,a)
pylab.text(0.5, 55, 'coeff. of correlation: ' + str(r2))
pylab.xlabel('real resistance')
pylab.ylabel('predicted resistance')

pylab.savefig(currentdir+'\\output\\YatchDesignResistanceRegression.jpg')
pylab.show()


# %%
model.save('Anomaly detection final model ')


