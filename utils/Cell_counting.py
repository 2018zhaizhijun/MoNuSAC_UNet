'''
    This code will generate an excel file with image name in rows and cell-count in columns

    Input
        data_path: Specify the path of downloaded images
    Output
        An excel file with name 'MoNuSAC-cell-count.xls' will store on the given data_path
'''


import os
# import openslide
# from xml.dom import minidom
# import numpy as np
# import openslide
# from openslide import open_slide
from glob import glob
# import cv2
# import matplotlib.pyplot as plt
# import scipy.io as sio
# from PIL import Image
# import scipy
# import scipy.ndimage
# from shapely.geometry import Polygon
# from skimage import draw
import xml.etree.ElementTree as ET
# import xlwt
from xlwt import Workbook

# Read svs files from the desired path
count = 0
data_path = 'D:\MoNuSAC_annotations' # Path to read data from
os.chdir(data_path)

patients = [x[0] for x in os.walk(data_path)] # Total patients in the data_path

label_map = {'Epithelial': 1,
             'Lymphocyte': 2,
             'Macrophage': 4,
             'Neutrophil': 3,
             }

image_count = 0

# Workbook is created
wb = Workbook()

# add_sheet is used to create sheet.
ccbt = wb.add_sheet('Cell count by type')
ccbt.write(0, 1, 'Epithelial')
ccbt.write(0, 2, 'Lymphocyte')
ccbt.write(0, 3, 'Neutrophil')
ccbt.write(0, 4, 'Macrophage')

for patient_loc in patients:

    # Read sub-images of each patient in the data path
    sub_images = glob(patient_loc+'/*.svs')
    for sub_image_loc in sub_images:
        image_count = image_count+1
        print(image_count)

        image_name = sub_image_loc
        ccbt.write(image_count, 0, sub_image_loc[len(patient_loc)+1:]) # Add image name in excel file
        if image_count > 1:
            ccbt.write(image_count-1,1, cell_count[1])
            ccbt.write(image_count-1,2, cell_count[2])
            ccbt.write(image_count-1,3, cell_count[3])
            ccbt.write(image_count-1,4, cell_count[4])

        #         Read xml file
        xml_file_name = image_name[:-4]
        xml_file_name = xml_file_name+'.xml'
        tree = ET.parse(xml_file_name)
        root = tree.getroot()
        cell_count = [0, 0, 0, 0, 0]

        # Generate binary mask for each cell-type
        for k in range(len(root)):
            label = [x.attrib['Name'] for x in root[k][0]]
            label = label[0]

            for child in root[k]:
                for x in child:
                    r = x.tag
                    if r == 'Attribute':
                        label = x.attrib['Name']

                    if r == 'Region':
                        cell_count[label_map[label]] = cell_count[label_map[label]]+1

ccbt.write(image_count, 1, cell_count[1])
ccbt.write(image_count, 2, cell_count[2])
ccbt.write(image_count, 3, cell_count[3])
ccbt.write(image_count, 4, cell_count[4])
wb.save('MoNuSAC-cell-count.xls')
