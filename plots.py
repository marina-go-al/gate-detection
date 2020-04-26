import numpy as np
import matplotlib.pyplot as plt
import pickle

# Inputs
fontSize = 25
figSize  = [8.5, 6]
lineWidth = 5
ROC_plot = 1
hst_plot = 1
size = 64
ROC_path = 'data/ROC_dataset/myROCs/'
deploy_path = 'data/report_images/'

# 1) ROC CURVES PLOT:

# Importing variables
x64      = np.load(ROC_path + 'x64.npy')
y64      = np.load(ROC_path + 'y64.npy')
x128     = np.load(ROC_path + 'x128.npy')
y128     = np.load(ROC_path + 'y128.npy')
x256     = np.load(ROC_path + 'x256.npy')
y256     = np.load(ROC_path + 'y256.npy')

# Printing variables
# print(x64)

# Plotting results
if ROC_plot == 1:
    plt.figure(num = 1, figsize = figSize)
    plt.plot(x64,  y64,  '-ob', linewidth = lineWidth)
    plt.plot(x128, y128, '-or', linewidth = lineWidth)
    plt.plot(x256, y256, '-og', linewidth = lineWidth)
    plt.xlabel("False Positive Rate", fontsize = fontSize)
    plt.ylabel("True Positive Rate", fontsize = fontSize)
    plt.legend(['64x64','128x128','256x256'],loc = 'best', fontsize = fontSize)
    plt.tick_params(axis = 'both', labelsize = fontSize)
    plt.xlim(0, 0.126)
    plt.ylim(0, 1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(deploy_path + 'ROC_curves.png')
    plt.show()

# 2) HISTORY (LOSS/ACCURACY) PLOT:
history_path = 'data/droneRace/outputWeights/'
if size == 64:
    history_name = 'drone_bs2_ep50_im272_size64_history.npy'
elif size == 128:  
    history_name = 'drone_bs2_ep25_im272_size128_history.npy'
elif size == 256:
    history_name = 'drone_bs2_ep15_im272_size256_history.npy'

# Importing variables
history = np.load(history_path + history_name,allow_pickle='TRUE').item()

# Plot training & validation accuracy/loss values
if hst_plot == 1:
    plt.figure(num = 2, figsize = figSize)
    plt.plot(history['accuracy'], linewidth = lineWidth)
    plt.plot(history['val_accuracy'], linewidth = lineWidth)
    #plt.title('Model accuracy', fontsize = fontSize+2)
    plt.ylabel('Accuracy', fontsize = fontSize)
    plt.xlabel('Epoch', fontsize = fontSize)
    plt.tick_params(axis = 'both', labelsize = fontSize)
    plt.legend(['Train', 'Test'], loc='best', fontsize = fontSize)
    plt.grid()
    plt.tight_layout()
    plt.savefig(deploy_path + 'accuracy_' + str(size) + '.png')
    plt.show()

    plt.figure(num = 3, figsize = figSize)
    plt.plot(history['loss'], linewidth = lineWidth)
    plt.plot(history['val_loss'], linewidth = lineWidth)
    #plt.title('Model loss', fontsize = fontSize+2)
    plt.ylabel('Loss', fontsize = fontSize)
    plt.xlabel('Epoch', fontsize = fontSize)
    plt.tick_params(axis = 'both', labelsize = fontSize)
    plt.legend(['Train', 'Test'], loc='best', fontsize = fontSize)
    plt.grid()
    plt.tight_layout()
    plt.savefig(deploy_path + 'loss_' + str(size) + '.png')
    plt.show()