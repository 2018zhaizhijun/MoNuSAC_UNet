# coding=gbk
import cv2
import glob
import os
# import staintools
import numpy as np
from patch_extractor import PatchExtractor


data_path = r'D:/MyData/MoNuSAC2020/CIA-Net/data/MoNuSeg' # Path to read data from
imgs_dir = data_path + r'/Images'
masks_dir = data_path + r'/Masks'
cnt_dir = data_path + r'/Contours'
save_imgs_dir = data_path + r'/UN_Images'
save_masks_dir = data_path + r'/UN_Masks'
save_cnt_dir = data_path + r'/UN_Contours'
os.mkdir(save_imgs_dir)
os.mkdir(save_masks_dir)
os.mkdir(save_cnt_dir)

imgs_list = glob.glob(imgs_dir + '/*.png')

# stain_normalizer = staintools.StainNormalizer(method='vahadane')

# path to target image
target_path = imgs_dir + r'/TCGA-21-5784-01Z-00-DX1.png'
target_img = cv2.imread(target_path)
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
# stain_normalizer.fit(target_img)
cnt = 0
R = []
G = []
B = []

for idx, img_path in enumerate(imgs_list):
    image_name = img_path[len(imgs_dir)+1:-4]
    print(image_name)

    img = cv2.imread(img_path)
    mask = cv2.imread((masks_dir + '/' + image_name + '_mask.png'), cv2.IMREAD_GRAYSCALE)
    contour =cv2.imread((cnt_dir + '/' + image_name + '_contour.png'), cv2.IMREAD_GRAYSCALE)
    img_patches, mask_patches, contour_patches = PatchExtractor().extract(img, mask, contour)

    for img, mask, contour in zip(img_patches, mask_patches, contour_patches):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = stain_normalizer.transform(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        R.append(np.average(img[:, :, 2]))
        G.append(np.average(img[:, :, 1]))
        B.append(np.average(img[:, :, 0]))
        cnt += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 定义结构元素的形状和大小
        # dst1 = cv2.erode(mask, kernel) # 腐蚀操作
        # dst2 = cv2.dilate(mask, kernel) # 膨胀操作
        # contour = dst2 - dst1
        contour = cv2.dilate(contour, kernel) # 膨胀操作
        cv2.imwrite("%s/%s.png" % (save_cnt_dir, cnt), contour)
        cv2.imwrite("%s/%s.png" % (save_imgs_dir, cnt), img)
        cv2.imwrite("%s/%s.png" % (save_masks_dir, cnt), mask)

BGR_mean = [np.mean(B), np.mean(G), np.mean(R)]
print(BGR_mean)
print("Total patches: " + str(cnt))
