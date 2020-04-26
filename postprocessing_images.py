from PIL import Image
import glob
import cv2
import config
import os

if config.POSTPROCESS_MASK == 1:
    path = 'data/droneRace/test/masks/*.png'
    deploy_path = 'data/ROC_dataset/mask_' + str(config.POSTPROCESSING_SIZE) + '/'
    if int(os.path.isdir(deploy_path)) == 0:
        print('Creating directory for postprocessed masks')
        os.mkdir(deploy_path)

if config.POSTPROCESS_PREDICTIONS == 1:
    path = 'data/droneRace/test/predict_ch1/*_predict.png'
    deploy_path = 'data/ROC_dataset/labeled' + str(config.INPUT_IMAGE_SIZE) + '_' + str(config.POSTPROCESSING_SIZE) + '/'
    if int(os.path.isdir(deploy_path)) == 0:
        print('Creating directory for postprocessed predictions')
        os.mkdir(deploy_path)

for filename in sorted(glob.glob(path)):
    word = filename.split('/')
    aux = cv2.imread(filename)
    auxResized = cv2.resize(aux,(config.POSTPROCESSING_SIZE,config.POSTPROCESSING_SIZE),interpolation = cv2.INTER_AREA)
    saveName = deploy_path + word[-1] #+ '_predict.png'
    cv2.imwrite(saveName, auxResized)
    #print(word[-1])

cv2.waitKey(0)