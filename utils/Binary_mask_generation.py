# coding=utf-8
import os
import numpy as np
from glob import glob
import cv2
from shapely.geometry import Polygon
from skimage import draw
import xml.etree.ElementTree as ET

# Read svs files from the desired path
count = 0
data_path = r'D:\MyData\MoNuSAC2020\CIA-Net\data\MoNuSeg' # Path to read data from
img_path = data_path + r'\Images'
anno_path = data_path + r'\Annotations'
# destination_path = data_path + r'\Masks' # Path to save binary masks corresponding to xml files

# try:
#     os.mkdir(destination_path)
# except OSError:
#     print("Creation of the mask directory %s failed" % destination_path)

images = glob(img_path + '\*.png')
annotations = glob(anno_path + '\.xml')

for image_loc in images:
    image_name = image_loc[len(img_path)+1:-4]
    print(image_name)

    img = cv2.imread(image_loc)

    # Read xml file
    xml_file_name = image_name+'.xml'
    tree = ET.parse(anno_path + os.path.sep + xml_file_name)
    root = tree.getroot()
    binary_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    contour = np.zeros(img.shape[:2], dtype=np.uint8)
    # binary_mask = Image.fromarray(binary_mask).convert('L')
    # contour = Image.fromarray(contour).convert('L')

    Regions = root[0][1].findall('Region')
    for region in Regions:
        regions = []
        vertices = region[1]
        coords = np.zeros((len(vertices), 2))
        # coords = []
        if len(vertices) < 3:
            continue
        for i, vertex in enumerate(vertices):
            coords[i][0] = vertex.attrib['X']
            coords[i][1] = vertex.attrib['Y']
            # coords.append((float(vertex.attrib['X']), float(vertex.attrib['Y'])))
        regions.append(coords)
        poly = Polygon(regions[0])
        # draw1 = ImageDraw.Draw(binary_mask)
        # draw1.polygon(coords, fill=255.0)
        # draw = ImageDraw.Draw(contour)
        # draw.polygon(coords, fill=0.0)
        vertex_row_coords = regions[0][:, 0]
        vertex_col_coords = regions[0][:, 1]
        fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords)
        fill_row_coords = fill_row_coords.clip(0, binary_mask.shape[0]-1)
        fill_col_coords = fill_col_coords.clip(0, binary_mask.shape[1]-1)
        binary_mask[fill_row_coords, fill_col_coords] = 255
        cv2.polylines(contour, np.int32([regions[0]]), 1, 255)

    mask_path = data_path + '\\Masks\\' + image_name + '_mask.png'
    cnt_path = data_path + '\\Contours\\' + image_name + '_contour.png'
    cv2.imwrite(mask_path, binary_mask)
    cv2.imwrite(cnt_path, contour)
