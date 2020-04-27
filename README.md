# 5044030-Marina-gateDetection

This repository contains all the necessary scripts and information to run the code corresponding to the *individial assignment* of the *AE4317 - Autonomous Flight of Micro Air Vehicles* course. The project is based on a binary image segmentation approach by means of the U-Net CNN. All the scripts have been extensively automatized so that eveyrything is easy to understand and execute just by changing some global variables.

**[IMPORTANT NOTE]:** In case that you would like to skip all the explanations, go directly to the [Workflow](#workflow) section to learn how to run the code.

## Files summary

In this section, a brief explanation of all files is provided:

### .py files to be run

`config.py`: where all global variables (for training, testing, preprocessing, postprocessing and plotting ROC curves) are defined. This is the file that matters most to you and you will be modifying most often.

`preprocessing_images.py`: file that preprocesses images for training, allocating the data set (original and label images) in the proper folders and with the correct format. Also, it already splits the data in two groups: training data (training set + validation set) and test set data.

`mainTrain.py`: execute this file for training.

`dataMine.py`: file utilized by `mainTrain.py` to handle the data and achieve a correct training.

`mainTest.py`: execute this file for testing.

`postprocessing_images.py`: file that converts the predictions after running `mainTest.py` from 1-channel images to 3-channel images and resizes them to the desired size to then be able to plot the ROC curves.

`ROC_curves.py`: file for plotting the ROC curve for a particular model.

`plots.py`: run this file to get the ROC plot and loss history plot from the report.

### .hdf5 and _history.npy files

The three `.hdf5` files located in `data/droneRace/outputWeights/` correspond to the parameters (weights) of the models developed for the project, for each of the three considered cases: 64 x 64, 128 x 128, 256 x 256, input image size. You will used these in case that you would like to reproduce the results enclosed in the presented report. Same applies for the `_history.npy` files, which correspond to the history loss and accuracy functions for each of the cases.

## Tree structure

This a list of all files and directories in the repo and how they are organized. <!--Some are not yet existing but will be automatically generated when running `preprocessing_images.py`, `mainTest.py`, `postprocessing_images.py` and `mainTrain.py`. This is, once you have followed at least once all the steps in [Workflow](#workflow).-->

```bash
.
├── config.py -> file where all global variables are defined
├── dataMine.py
├── getMinSize.py
├── mainTest.py
├── mainTrain.py
├── modelMine.py
├── plots.py
├── postprocessing_images.py
├── preprocessing_images.py
├── ROC_curves.py
├── data
│   ├── droneRace
│   │   ├── outputWeights -> weight and history loss files are stored here
│   │   │   ├── drone_bs2_ep15_im272_size256.hdf5
│   │   │   ├── drone_bs2_ep15_im272_size256_history.npy
│   │   │   ├── drone_bs2_ep25_im272_size128.hdf5
│   │   │   ├── drone_bs2_ep25_im272_size128_history.npy
│   │   │   ├── drone_bs2_ep50_im272_size64.hdf5
│   │   │   └── drone_bs2_ep50_im272_size64_history.npy
│   │   ├── test -> test set data is stored here
│   │   │   ├── masks
│   │   │   ├── original
│   │   │   └── predict_ch1
│   │   └── train -> training set data is stored here
│   │       ├── aug -> where data for training is automatically deployed with the required naming
│   │       ├── image
│   │       └── label
│   ├── ROC_dataset -> data to plot the ROC curves will be generated and stored here
│   └── WashingtonOBRace -> folder that you have to replace with your own data set
└── myconfig
    ├── config128.txt -> Copy this file into config.py to reproduce 128 x 128 results
    ├── config256.txt -> Copy this file into config.py to reproduce 256 x 256 results
    └── config64.txt  -> Copy this file into config.py to reproduce  64 x  64 results
```

## Workflow


**1)**: Substitute the *WashingtonOBRace* example directory by your own *WashingtonOBRace* folder containing the data set: the INNER one directly containing the data set, (NOT the OUTER one that contains the README.txt and the inner directory). So remeber, the INNER FOLDER (exactly the same that we were provided on Brightspace: do not change names of files or the directory)

**2)**: Run `preprocessing_images.py`

Now, choose between **A** and **B**:

### A. Reproducing the results from the report (points 3 and 4):

Now, to reproduce the results for the 128 x 128 case from the report (same applies to 64 x 64 or 256 x 256):

**3)** Copy the content of `myconfig/config64.txt` into `config.py`

**4)** Run `mainTest.py` and check out the results that have been deployed in `data/droneRace/test/predict_ch1/`

### B. Training your own model and generating your own results:

**3.1)** Customize your parameters in `config.py`

**3.2)** Run `mainTrain.py` and wait until the training has finished

**4)** Run `mainTest.py` and check out the results that have been deployed in `data/droneRace/test/predict_ch1/`

### Plotting ROC curves:

**5)** Run `postprocessing_images.py`

**6)** Run `ROC_curves.py` and check out the results
