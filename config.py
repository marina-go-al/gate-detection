
# DEFINE GLOBAL VARIABLES IN THIS FILE

'''
#### IMPORTANT NOTE! -> The current configuration is the same as config128.txt (the 128 x 128 case in the report). 
####                    So careful!! Do not train the network again with this configuration or the current weights file (the current model) will be overwritten
####                    If this happens, you can redownload the corresponding weights file again from GitHub
'''

REPRODUCE_ORIGINAL_RESULTS = 1 # Set this value to 1 in case that you would like to split the dataset exactly as it originally was to test the results in the report
INPUT_IMAGE_SIZE = 128         # Define input image size (64, 128, 256)
TEST_SET_IMAGES  = 46          # Total number of test set images in data/droneRace/test/original

# Variables for training and testing (making predictions):
BATCH_SIZE = 2              # Batch size
EPOCH      = 25             # Number of epoch
VALIDATION_SPLIT = 0.15     # Validation split percentage
GEN_IMAGE_NUM = 16          # See below
AUX_TOTAL = GEN_IMAGE_NUM**2 + GEN_IMAGE_NUM # Number of images to be prepared for training
                                             # Make sure that AUX_TOTAL > (training set + validation set)
WEIGHTS_HISTORY_SAVE_NAME = 'drone_bs' + str(BATCH_SIZE) + '_ep' + str(EPOCH) + '_im' + str(AUX_TOTAL) + '_size' + str(INPUT_IMAGE_SIZE) # Name for saving wrights and history
GEN_DATA = 1                # Generate data for training. 1: YES, 0: NO. Better set to 1.

# Variables exclusively for testing:
SET_THRESHOLD = 0           # 0: Grayscale predictions, 1: Binary predictions
THRESHOLD = 0.2             # Define threshold in case that SET_THRESHOLD == 1 for different results 

# Postprocessing size for the ROC curves or better visualization
POSTPROCESSING_SIZE = 315   # Size of the post_processed image
POSTPROCESS_MASK = 1        # 1: Masks are postprocessed, 0: Masks are not post processed (Always set to one the first time you plot a ROC curve)
POSTPROCESS_PREDICTIONS = 1 # 1: Predictions are postprocessed, 0: Predictions are not post processed (Always set to one the first time you plot a ROC curve)
