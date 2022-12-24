import matplotlib.pyplot as plt
import numpy as np


def PLOT_HISTORY(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('MAE')
  plt.plot(history.epoch, np.array(history.history['mae']), 
           label='Train')
  plt.plot(history.epoch, np.array(history.history['val_mae']),
           label = 'Val')
  plt.legend()
  plt.ylim([0,max(history.history['val_mae'])])