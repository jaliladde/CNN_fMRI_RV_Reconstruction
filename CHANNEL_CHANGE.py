import numpy as np

def CHANNEL_CHANGE(data_input_window, window_size, num_ROIs):

    data_input_window_CL=np.zeros((data_input_window.shape[0],window_size,num_ROIs))   # reshape the data for Channel Last
    
    for i in range(data_input_window.shape[0]):
        data_temp=data_input_window[i,:,:]
        data_input_window_CL[i,:,:]=np.transpose(data_temp)
        
    return data_input_window_CL