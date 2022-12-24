import numpy as np

def WINDOWING(data_input, Volumes, window_size):

    hcp_w=[]
    
    for i in data_input:
      for j in range(0,Volumes-window_size+1):          # window
        hcp_w.append(i[:, j:window_size+j])
    
    hcp_w=np.array(hcp_w) 
    data_input_window=hcp_w                             # data after applying moving window 
    return data_input_window
