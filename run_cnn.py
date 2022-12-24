import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint



from R2 import R2
from PLOT_HISTORY import PLOT_HISTORY
from NORMALIZE_INPUT import NORMALIZE_INPUT
from NORMALIZE_TARGET import NORMALIZE_TARGET
from WINDOWING import WINDOWING
from CHANNEL_CHANGE import CHANNEL_CHANGE
from SAVE_DATA_TEST import SAVE_DATA_TEST
from SAVE_DATA_TRAIN import SAVE_DATA_TRAIN
from SPLIT_DATA import SPLIT_DATA
from SPLIT_DATA_SEMI_RANDOM import SPLIT_DATA_SEMI_RANDOM
from BUILD_MODEL import BUILD_MODEL
from FIT_MODEL import FIT_MODEL
# \............................................................................
    # set the initial parameters
    # load data  
    # process input data
    # apply window to the BOLD signals
    # design the CNN network
    # train the model using TRAINING subset and monitor its performance on VALIDATION subset
    # plot training & convergence process
    # save the trained model
    # evaluate the performance on TEST subset
# ............................................................................/

# SET THE INITIAL PARAMETERS __________________________________________________

data_path='/path_of_data/'
main_path='/path_of_code_files/'


num_ROIs=90
Volumes=478                                    # In HCP-D project, the number of fMRI volumes are 478
window_size=65                                 # change it according to the desired window size [9 , 17 , 33 , 65 , 129 , 257]
num_epochs=300                     
window_start_point=int((window_size/2)-1)         
window_end_point=-window_start_point
BatchSize=64
n_samples=352                     # from all samples, randomly select a subset of them 
number_folds = 10
number_test_each_fold = 35

n_col = 1

# LOAD DATA AND NORMALIZE IT __________________________________________________

inputs=np.load(data_path + 'Input_Data.npy')
targets=np.load(data_path + 'RV.npy')

shuffle_scans=np.random.permutation(inputs.shape[0])
# =============================================================================
# sub=sub[0:n_samples]     
# sub = np.sort(sub)                                 # from all samples, randomly select a subset of them
# =============================================================================
data_input=inputs[shuffle_scans, :, :]
data_target_temp=targets[:, shuffle_scans]
# PROCESS INPUT DATA __________________________________________________________

data_input_normalized=NORMALIZE_INPUT(data_input)
data_input_window=WINDOWING(data_input_normalized, Volumes, window_size)
data_input_window_CL=CHANNEL_CHANGE(data_input_window, window_size, num_ROIs)

# PROCESS TARGET DATA _________________________________________________________

data_target=NORMALIZE_TARGET(data_target_temp, window_start_point, window_end_point)

# SPLIT THE DATASET ___________________________________________________________
List_Test_Scan = []                                     # create a list [10 x 35]. The last row will have 37 elements as we have 352 scans.
                                                        # each row will be used as the TEST scans in each run. 
                                                        # we have 10 TEST subset, which each of them includes 35 unseen scans in the training process

for i in range(number_folds):
    temp_list=list(range(i*number_test_each_fold, (i+1)*number_test_each_fold, 1))
    if i==9:
        temp_list=list(range(i*number_test_each_fold, n_samples, 1))
    List_Test_Scan.append(temp_list)
    
# FIT THE MODEL _______________________________________________________________

for fold_k in range(number_folds):
    train_inputs, valid_inputs, test_inputs, train_targets, valid_targets, test_targets = SPLIT_DATA(data_input_window_CL, data_target, List_Test_Scan, fold_k, n_samples)                  


    # DESIGN THE CNN NETWORK ______________________________________________________
    model=BUILD_MODEL(train_inputs)
    # TRAIN THE MODEL _____________________________________________________________
    history=FIT_MODEL(model, train_inputs, train_targets, valid_inputs, valid_targets, 
              num_epochs, BatchSize, main_path)
    
    # PLOT TRAINING & CONVERGENCE PROCESS _________________________________________
    
    model_history = history.history
    print(type(model_history))
    print(model_history.keys())          # dict_keys(['loss', 'mae', 'r2', 'val_loss', 'val_mae', 'val_r2'])
    
    PLOT_HISTORY(history)

    # EVALUATE THE PERFORMANCE ON TETS DATA _______________________________________
    output =model.predict(test_inputs, batch_size=BatchSize)
    
    # test_corr='%.2f'%test_corr
    # print('correlation test is:   ', test_corr)
    
    # SAVE ________________________________________________________________________
    SAVE_DATA_TEST(main_path, test_targets, output, fold_k)



