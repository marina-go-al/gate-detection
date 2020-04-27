from modelMine import *
from dataMine import *
import matplotlib.pyplot as plt
import config
import os

# INPUTS
name           = config.WEIGHTS_HISTORY_SAVE_NAME    # Name for the weights and history files to be saved
gen_data       = config.GEN_DATA                     # Generate data for training. 1: YES, 0: NO
gen_num_images = config.GEN_IMAGE_NUM                # Specify the number of data to be generated. Currently, 16 -> 16*16 + 16 = 272, 
                                                     # with 10 repeated that will not be used. Therefore, 262 in total
save_path      = 'data/droneRace/outputWeights/'     # Path where both the weights and history data from training will be saved
'''
The current notation for the name is:
name = 'drone_bs#{batch_size}_ep#{epoch}_im#{image_generation}_size#{input_image_size}
'''

if gen_data == 1:

    # Uncomment this part for data augmentation purposes
    '''
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    '''
    data_gen_args = dict()

    deployPath = 'data/droneRace/train/aug'

    if int(os.path.isdir(deployPath)) == 0:
        print('Creating aug directory for training')
        os.mkdir(deployPath)
  
    myGene = trainGenerator(gen_num_images,'data/droneRace/train/','image','label',data_gen_args,save_to_dir = deployPath) # The training data will be saved to save_to_dir location
    num_batch = gen_num_images
    for i,batch in enumerate(myGene):
        if(i >= num_batch):
            break

image_arr,mask_arr = geneTrainNpy("data/droneRace/train/aug/","data/droneRace/train/aug/")
#np.save("data/image_arr.npy",image_arr)
#np.save("data/mask_arr.npy",mask_arr)

model = unet()
model_checkpoint = ModelCheckpoint(save_path + name + '.hdf5', monitor='loss',verbose=1, save_best_only=True)

history = model.fit(image_arr, mask_arr, batch_size=config.BATCH_SIZE, epochs=config.EPOCH, verbose=1,validation_split=config.VALIDATION_SPLIT, shuffle=True, callbacks=[model_checkpoint])
np.save(save_path + name + '_history.npy',history.history)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()