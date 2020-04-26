from modelMine import *
from dataMine import *
import matplotlib.pyplot as plt
import pickle
import timeit
from keras.utils import plot_model
import config

# INPUTS

num              = config.TEST_SET_IMAGES             # Total number of test set images.
set_threshold    = config.SET_THRESHOLD               # 0: grayscale image, 1: binary image
threshold        = config.THRESHOLD                   # Select 0 < threshold < 1 for the conversion to binary image in case that set_threshold == 1
testSet_path     = "data/droneRace/test/original"     # Path where the test set image are stored
deploy_path      = "data/droneRace/test/predict_ch1"  # Path where predictions will be deployed
name             = config.WEIGHTS_HISTORY_SAVE_NAME   # Name of weights and history data files to make the predictions

plot_history     = 1                                  # Set 1 for plotting the history and 0 otherwise

if plot_history == 1:
    history = np.load('data/droneRace/outputWeights/' + name + '_history.npy',allow_pickle='TRUE').item()

    # Plot training & validation accuracy values
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


testGene = testGenerator(testSet_path)
model = unet()
model.load_weights("data/droneRace/outputWeights/" + name + ".hdf5")

tic = timeit.default_timer()
results = model.predict_generator(testGene,num,verbose=1)
toc = timeit.default_timer()

elapsed_time = toc-tic
print(str(elapsed_time) + ' seconds elapsed for computing the prediction of ' + str(num) + ' images from the test set.')

if set_threshold == 0:
    saveResult(deploy_path,results)
elif set_threshold == 1:
    myfilter = threshold * np.ones(results.shape)
    results = (results > myfilter)
    saveResult(deploy_path,results)