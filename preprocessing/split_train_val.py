import glob
import numpy as np
import os
from shutil import copy


data_path = r'D:/MyData/MoNuSAC2020/CIA-Net/data/MoNuSeg'
img_path = data_path + r'/UN_Images'
mask_path = data_path + r'/UN_Masks'
cnt_path = data_path + r'/UN_Contours'

train_size = 0.8

images = glob.glob(img_path + r'/*.png')
data_num = len(images)
train_num = int(data_num*train_size)
data_idx = np.arange(1, data_num+1)
np.random.shuffle(data_idx)

train_idx = data_idx[:train_num]
val_idx = data_idx[train_num:]

train_dir = data_path + r"/TRAIN_UN"
if os.path.exists(train_dir):
    os.remove(train_dir)
os.mkdir(train_dir)
os.mkdir(train_dir + r"/Image")
os.mkdir(train_dir + r"/Mask")
os.mkdir(train_dir + r"/Contour")

for cnt, idx in zip(np.arange(train_num), train_idx):
    file_from = '/' + str(idx) + '.png'
    file_to = '/' + str(cnt) + '.png'
    copy(img_path + file_from, train_dir + r"/Image" + file_to)
    copy(mask_path + file_from, train_dir + r"/Mask" + file_to)
    copy(cnt_path + file_from, train_dir + r"/Contour" + file_to)

val_dir = data_path + r"/VAL_UN"
if os.path.exists(val_dir):
    os.remove(val_dir)
os.mkdir(val_dir)
os.mkdir(val_dir + r"/Image")
os.mkdir(val_dir + r"/Mask")
os.mkdir(val_dir + r"/Contour")

for cnt, idx in zip(np.arange(data_num-train_num), val_idx):
    file_from = '/' + str(idx) + '.png'
    file_to = '/' + str(cnt) + '.png'
    copy(img_path + file_from, val_dir + r"/Image" + file_to)
    copy(mask_path + file_from, val_dir + r"/Mask" + file_to)
    copy(cnt_path + file_from, val_dir + r"/Contour" + file_to)

