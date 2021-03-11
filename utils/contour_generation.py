import os
from glob import glob
import cv2
import scipy.io
import numpy as np
from config import IMAGE_SIZE


data_path = r'D:\MyData\MoNuSAC2020\CIA-Net\data\MoNuSeg' # Path to read data from

patients = [x[0] for x in os.walk(data_path)] # Total patients in the data_path
labels = ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]

for patient_loc in patients:
    patient_name = patient_loc[len(data_path)+1:] # Patient name
    print(patient_name)

    # To make patient's name directory in the destination folder
    try:
        os.mkdir(patient_name)
    except OSError:
        print("\n Creation of the patient's directory %s failed" % patient_name)

    # Read sub-images of each patient in the data path
    sub_images = glob(patient_loc+'/*.svs')
    for sub_image_loc in sub_images:
        sub_image_name = sub_image_loc[len(data_path)+len(patient_name)+1:-4]
        print(sub_image_name)

        # To make sub_image directory under the patient's folder
        sub_image_dir = './'+patient_name+'/'+sub_image_name # Destination path
        image_name = sub_image_loc
        masks = []
        whole_mask = np.zeros(IMAGE_SIZE)

        for label in labels:
            mask_dir = sub_image_dir + '/' + label
            sub_mask = glob(mask_dir+'/*.tif')[0]
            mask = cv2.imread(sub_mask, 1)
            assert mask.shape == IMAGE_SIZE, "mask shape doesn't match image shape"
            masks += mask
            whole_mask += mask

        assert len(masks) == 4, "Something wrong with the number of masks"
        scipy.io.savemat(sub_image_dir+"masks.mat",
                         {'0': masks[0], '1': masks[1], '2': masks[2], '3': masks[3]})

        cv2.imwrite(sub_image_dir+"whole_mask.jpg", whole_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 定义结构元素的形状和大小
        dst1 = cv2.erode(whole_mask, kernel) # 腐蚀操作
        dst2 = cv2.dilate(whole_mask, kernel) # 膨胀操作
        contour = dst2 - dst1
        cv2.imwrite(sub_image_dir+"contour.png", contour)
