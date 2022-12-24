import numpy as np

def NORMALIZE_INPUT(data_input):
    t=[]
    for i in data_input:
      z=((i.T-i.mean(axis=1))/i.std(axis=1)).T          # normalization
      t.append(z)

    data_input=np.array(t)
    
    return data_input
    print('shape input after normalization: ',data_input.shape)
