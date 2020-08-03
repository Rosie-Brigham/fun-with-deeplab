import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import numpy as np

import os, shutil

# palette (color map) describes the (R, G, B): Label pair
palette = {(0, 0, 0) : 0 ,
            (255, 255, 255) : 1 #water
          }

def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d


label_dir = 'deeplab/datasets/PQR/SegmentationClass/'
new_label_dir = 'deeplab/datasets/PQR/SegmentationClassRaw/'

if not os.path.isdir(new_label_dir):
	print("creating folder: ",new_label_dir)
	os.mkdir(new_label_dir)
else:
	print("Folder alread exists. Delete the folder and re-run the code!!!")


label_files = os.listdir(label_dir)
# needed for local, but not colab apparently
if '.DS_Store' in label_files:
    label_files.remove('.DS_Store')

for l_f in tqdm(label_files):
    arr = np.array(Image.open(label_dir + l_f))
    # import code; code.interact(local=dict(globals(), **locals()))
    arr = arr[:,:,0:3]
    arr_2d = convert_from_color_segmentation(arr)
    Image.fromarray(arr_2d).save(new_label_dir + l_f)