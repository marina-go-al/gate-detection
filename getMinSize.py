import glob
import cv2

shape_dim1 = []
shape_dim2 = []
count = 0
path = 'data/droneRace/train/misc/size_label/*.png'

for filename in glob.glob(path):

    aux = cv2.imread(filename)
    mask_shape = aux.shape
    shape_dim1.append(mask_shape[0])
    shape_dim2.append(mask_shape[1])

print(min(shape_dim1))
print(min(shape_dim2))

cv2.waitKey(0)