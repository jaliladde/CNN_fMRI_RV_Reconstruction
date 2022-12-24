import numpy as np

def NORMALIZE_TARGET(data_target_temp, window_start_point, window_end_point):

    data_target=data_target_temp[window_start_point:window_end_point,:]
    print('shape target after windowing: ',data_target.shape)
    
    data_target=data_target.reshape(-1,1,order='F')
    data_target=(data_target-data_target.mean())/data_target.std()             # normaliz target
    print('shape target after normalization: ',data_target.shape)
    
    return data_target