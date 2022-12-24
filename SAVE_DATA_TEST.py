import numpy as np


def SAVE_DATA_TEST(main_path, test_targets, output, fold_k):
    
    save_path = main_path + 'saved_data/'
    
    np.save(save_path + 'test_targets_' + str(fold_k) +'.npy', test_targets)
    np.save(save_path + 'test_output' + str(fold_k) +'.npy', output)
    
