import glob
import pickle
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch


image_list = []
num_images = input('Number of images (30):')
num_images = int(num_images)

foldername = input('Images folder name (Train_images)')
foldername = foldername + '/'
image_format = input('Image format (jpg)')
image_format = '*.' + image_format

basewidth = input('Resiezed images horisontal dimension (100):')
basewidth = int(basewidth)

hsize = input('Resiezed images vertical dimension (100):')
hsize = int(hsize)

save_file = input('Save file name:')
save_file = save_file + '.pkl'

# img = Image.open('training_images_table1/1.jpg')
# wpercent = (basewidth / float(img.size[0]))
# hsize = int((float(img.size[1]) * float(wpercent)))
image_data = np.zeros((num_images, 1,  basewidth, basewidth))

img_index = 0
for filename in sorted(glob.glob(foldername + image_format)):
    img = Image.open(filename)
    img_resize = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img_resize_np = np.array(img_resize)
    img_bw = img_resize_np[:, :, 0] * 0.299 + img_resize_np[:, :, 1] * 0.587 + img_resize_np[:, :, 2] * 0.114
    image_data[img_index, :, :, :] = img_bw
    img_index = img_index + 1

output = open(save_file, 'wb')
pickle.dump(image_data, output)
output.close()
