import random
import numpy as np

def encode(df):
    df_out = df[0]
    df_in = df.drop([0],axis=1)

    normalized_df_in=df_in/255


    input = normalized_df_in.values.tolist()

    def oneHot(value):
        lst=[]
        for a in range(10):
            if a==value:
                lst.append(1)
            else:
                lst.append(0)
        return lst

    output = [oneHot(value) for value in df_out.values.tolist()]

    return input, output

def decode(input):
    result=[]
    for row in input:
        for i2,value in enumerate(row.tolist()):
            
            if value == 1:
                result.append(i2)
                #break
            
    return result

def getAccuracy(model):
    output=decode(model.output)

    for i in range(len(output)):
        if output[i]==np.argmax(model.layers[-1].a[i]):
            output[i]=1
        else:
            output[i]=0
    
    return str(100*sum(output)/len(output)) + ' %'
    

def sample(a,b,N=10):
	c=random.sample(range(len(a)),N)

	anew=[]
	bnew=[]

	for e in c:
		
		anew.append(a[e])
		bnew.append(b[e])
	
	return anew,bnew
