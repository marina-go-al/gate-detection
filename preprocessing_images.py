from PIL import Image
import glob
import cv2
import config
import os
import numpy as np
import random
import config

path = 'data/WashingtonOBRace/'
image_prefix = 'img_'
mask_prefix = 'mask_'
image_list = sorted(glob.glob(os.path.join(path,"%s*.png"%image_prefix)))
mask_list  = sorted(glob.glob(os.path.join(path,"%s*.png"%mask_prefix)))
num_testSet = config.TEST_SET_IMAGES

if config.REPRODUCE_ORIGINAL_RESULTS == 0:
    index_list = [aux.split('_')[1] for aux in mask_list]
    index_list = [aux.split('.')[0] for aux in index_list]
    index_list = random.sample(index_list,len(index_list))
    index_list = index_list[0:num_testSet]
else:
    index_list = [12,22,28,51,57,64,71,115,124,138,144,160,163,165,179,185,192,197,232,235,239,246,254,260,273,277,280,292,297,299,321,324,328,385,390,396,399,404,408,411,414,417,420,431,433,437]

test_list = [('img_' + str(j) + '.png') for j in index_list] # Original test set (as in the report)

#index_list = []
#print(index_list)
#print(len(index_list))

image_deployPath = 'data/droneRace/train/image/'
masks_deployPath = 'data/droneRace/train/label/'
testImage_deployPath  = 'data/droneRace/test/original/'
testMask_deployPath  = 'data/droneRace/test/masks/'

train_deployPath = 'data/droneRace/train/'
test_deployPath = 'data/droneRace/test/'

if int(os.path.isdir(train_deployPath)) == 0:
        print('Creating directory for training data')
        os.mkdir(train_deployPath)
if int(os.path.isdir(test_deployPath)) == 0:
        print('Creating directory for testing data')
        os.mkdir(test_deployPath)
if int(os.path.isdir(image_deployPath)) == 0:
        print('Creating directory for training images')
        os.mkdir(image_deployPath)
if int(os.path.isdir(masks_deployPath)) == 0:
        print('Creating directory for training masks')
        os.mkdir(masks_deployPath)
if int(os.path.isdir(testImage_deployPath)) == 0:
        print('Creating directory for testing images')
        os.mkdir(testImage_deployPath)
if int(os.path.isdir(testMask_deployPath)) == 0:
        print('Creating directory for testing images')
        os.mkdir(testMask_deployPath)

if int(len(image_list) == len(mask_list)) == 0:
    print('The number of IMAGES and MASKS is not the same. Review your dataset and fix before continue!!!')

countTest = 0
countRest = 0

for i in range(0,len(image_list)):
    image_path = image_list[i]
    mask_path = mask_list[i]
    word_image = image_path.split('/')
    word_mask = mask_path.split('/')
    saveName_image = image_deployPath + word_image[-1]
    saveName_mask =masks_deployPath +  word_mask[-1]

    im = cv2.imread(image_path)
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(mask_path)
    maskGray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if word_image[-1] in test_list:
        cv2.imwrite(testImage_deployPath + str(countTest) + '.png',imGray)
        cv2.imwrite(testMask_deployPath + str(countTest) + '.png',maskGray)
        countTest += 1
    else:
        cv2.imwrite(saveName_image,imGray)
        cv2.imwrite(saveName_mask,maskGray)
        countRest += 1

#print(index_list)
print(countTest)
print(countRest)