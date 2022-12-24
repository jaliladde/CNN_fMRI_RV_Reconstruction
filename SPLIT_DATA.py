import numpy as np
from sklearn.model_selection import train_test_split


def SPLIT_DATA(x_input, y_input, List_Test_Scan, fold_k, n_samples):
    
    all_scans = np.arange(0,n_samples)
    test_scans =  np.array( List_Test_Scan[fold_k])
    train_valid_scans = np.delete(all_scans , test_scans)
    

    
    Start_Point = 415 * test_scans[0]                       # As we loss 65 timepoints, from 478 volumes we can estimate 415 timepoints
    End_Point = 415 * test_scans[-1]
    test_inputs = x_input[Start_Point:End_Point, :, :]
    test_targets = y_input[Start_Point:End_Point]
    
    
    
    
    
    test_scans_for_delete = np.arange(Start_Point,End_Point)
    
    train_valid_inputs = x_input
    train_valid_targets = y_input
    
    train_valid_inputs = np.delete(train_valid_inputs , test_scans_for_delete, axis = 0)
    train_valid_targets = np.delete(train_valid_targets , test_scans_for_delete, axis = 0)
    
    
    train_inputs, valid_inputs, train_targets, valid_targets = train_test_split(train_valid_inputs, 
                                                                          train_valid_targets, 
                                                                          test_size=0.2, 
                                                                          shuffle=True)
    
    
    
    return train_inputs, valid_inputs, test_inputs, train_targets, valid_targets, test_targets