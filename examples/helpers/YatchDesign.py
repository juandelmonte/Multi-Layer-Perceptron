import random

def encode(df, output_v):
    df_out = df[output_v].copy()
    df_in = df.drop([output_v],axis=1).copy()

    input = df_in.values.tolist()
    output = df_out.values.tolist()

    return input, output

def decode(input):
    result=[]
    for row in input:
        for i2,value in enumerate(row.tolist()):
            
            if value == 1:
                result.append(i2)
                #break
            
    return result

def generate_samples(input, output, k=5):
    input_training, output_training, input_test, output_test = [],[],[],[]
    l= len(input)
    a=random.sample(range(l),l//k)

    for b in range(l):
        if b in a:    
            input_test.append(input[b])
            output_test.append(output[b])
        else:
            input_training.append(input[b])
            output_training.append(output[b])
    
    return input_training, output_training, input_test, output_test