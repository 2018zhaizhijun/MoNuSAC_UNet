import glob
import numpy as np
import random
import cv2
from config import BGR_mean, INPUT_SIZE, MASK_SIZE


def train_gen(data_dir, batch_size=10, rand=False, augs=None):
    files = glob.glob(data_dir + r'/Image/*.png')
    data_num = len(files)
    idx = 0

    if rand:
        random.seed(0)
        idx = random.randint(0, data_num-1)

    while True:
        im_batch = []
        mask_batch = []

        for i in range(batch_size):
            im = cv2.imread(data_dir + '/Image/' + str(idx) + '.png')
            mask = cv2.imread(data_dir + '/Mask/' + str(idx) + '.png', 0)
            contour = cv2.imread(data_dir + '/Contour/' + str(idx) + '.png', 0)
            data = np.array(im).astype('float64')
            label1 = np.array(mask) / 255.0
            label2 = np.array(contour)

            # data augmentation
            for aug in augs:
                data, label1, _ = aug.apply(data, label1, label2)
            data = (data - np.array(BGR_mean, dtype='float64')) / 255.0
            # data -= np.array(BGR_mean/255).astype('float64')
            #  width x height x channel

            im_batch.append(data)
            mask_batch.append(label1)

            if rand:
                idx = random.randint(0, data_num-1)
            else:
                idx += 1
            if idx == data_num:
                idx = 0

        im_batch = np.array(im_batch).reshape((batch_size, *im_batch[0].shape))
        mask_batch = np.array(mask_batch).reshape((batch_size, *MASK_SIZE))
        yield (im_batch, mask_batch)


def val_gen(data_dir, batch_size=10):
    files = glob.glob(data_dir + r'/Image/*.png')
    data_num = len(files)
    idx = 0

    while True:
        im_batch = []
        mask_batch = []

        for i in range(batch_size):
            im = cv2.imread(data_dir + '/Image/' + str(idx) + '.png')
            mask = cv2.imread(data_dir + '/Mask/' + str(idx) + '.png', 0)
            data = np.array(im).astype('float64')
            label1 = np.array(mask) / 255.0

            data = (data - np.array(BGR_mean, dtype='float64')) / 255.0
            # data -= np.array(BGR_mean/255).astype('float64')

            im_batch.append(data)
            mask_batch.append(label1)

            idx += 1
            if idx == data_num:
                idx = 0

        im_batch = np.array(im_batch).reshape((batch_size, *im_batch[0].shape))
        mask_batch = np.array(mask_batch).reshape((batch_size, *MASK_SIZE))
        yield (im_batch, mask_batch)
