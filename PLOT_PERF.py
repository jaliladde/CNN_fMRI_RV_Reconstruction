import numpy as np
import matplotlib.pyplot as plt

def PLOT_PERF(measured, recon, Metrics, n_signals, n_plots, dataset_name):
    
    rand_signal=np.random.permutation(n_signals)
    rand_signal=rand_signal[0:n_plots]
        
    plt.figure()
    for i in range(n_plots):
        plt.subplot(n_plots,1, i+1)
        plt.plot(measured[:,rand_signal[i]],'r-', label='measured')
        plt.plot(recon[:,rand_signal[i]],'b-', label='reconstructed')
        plt.legend(loc="upper right", fontsize=12)
        plt.title('r: ' + str(Metrics[rand_signal[i],0]) + ' ,   mse: ' + str(Metrics[rand_signal[i],1]),  
                  fontsize=14,fontname='Times New Roman',fontweight='bold',fontstyle='italic', color='k')
        plt.xticks([])
        if i+1==n_plots:
            plt.xticks([10, 100, 200, 300])
        
        plt.suptitle(dataset_name + '  dataset', fontsize=22,fontname='Times New Roman',fontweight='bold', 
                   color='y')
    
    