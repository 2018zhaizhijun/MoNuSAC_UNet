'''
    This code will generate an excel file containing multiple sheets (named as participants team-names). Each sheet will save image names in rows and respective PQ metrics in columns.
    Epithelial = column 1, Lymphocyte = column 2, Macrophage = column 3, Neutrophil = column 4
    Note: This code will work if n-ary masks are stored in both ground truth and predicted path. If the mask is stored as binary, it will first convert it into n-ary and then compute PQ metric
    Please run n-ary mask generation code from here to see ground truth and predicted masks format.
    The code will save masks to compute PQ metric as given below:
    -Folder -> Patient name
    -Sub-folder -> Sub-images under each patient
    -Sub-Sub-folder -> Annotated cell-type on each sub-image which contains n-ary masks (saved as mat file)

    Input
        ground_truth_path: Path to read ground truth masks from
        Predicted_path: Path to read participant folders from
    Output
        An excel file with name MoNuSAC-testing-PQ.xls will store on the given ground_truth_path
'''


# import os
import numpy as np
import glob
# import cv2
import scipy.io as sio
# from PIL import Image
# import scipy
import scipy.ndimage
# import xlwt
from xlwt import Workbook


# Compute Panoptic quality metric for each image
def Panoptic_quality(ground_truth_image, predicted_image):
    TP = 0
    FP = 0
    FN = 0
    sum_IOU = 0
    matched_instances = {}
    # Create a dictionary to save ground truth indices in keys and predicted matched instances as velues
    # It will also save IOU of the matched instance in [indx][1]

    # Find matched instances and save it in a dictionary
    for i in np.unique(ground_truth_image):
        if i == 0:
            pass
        else:
            temp_image = np.array(ground_truth_image)
            temp_image = temp_image == i
            matched_image = temp_image * predicted_image

            for j in np.unique(matched_image):
                if j == 0:
                    pass
                else:
                    pred_temp = predicted_image == j
                    intersection = sum(sum(temp_image*pred_temp))
                    union = sum(sum(temp_image + pred_temp))
                    IOU = intersection/union
                    if IOU > 0.5:
                        matched_instances[i] = j, IOU

    # Compute TP, FP, FN and sum of IOU of the matched instances to compute Panoptic Quality

    pred_indx_list = np.unique(predicted_image)
    pred_indx_list = np.array(pred_indx_list[1:])

    # Loop on ground truth instances
    for indx in np.unique(ground_truth_image):
        if indx == 0:
            pass
        else:
            if indx in matched_instances.keys():
                pred_indx_list = np.delete(pred_indx_list, np.argwhere(pred_indx_list == [indx][0]))
                TP = TP+1
                sum_IOU = sum_IOU+matched_instances[indx][1]
            else:
                FN = FN+1
    FP = len(np.unique(pred_indx_list))
    PQ = sum_IOU/(TP+0.5*FP+0.5*FN)

    return PQ


# Ground truth path to read data from
ground_truth_path = 'D:\MoNuSAC\Submissions\Mask_files\Organizers_MoNuSAC_test_results'
# Path to read predicted outcomes from
Predicted_path = 'D:\MoNuSAC\Submissions\Mask_files\Predicted_updated_masks'


import os
os.chdir(ground_truth_path)
participants_folders = glob.glob(Predicted_path+'/**')

cell_types = ['Epithelial', 'Lymphocyte', 'Macrophage', 'Neutrophil']
# Ground Truth files
files = glob.glob('./**/**')

# Workbook is created
wb = Workbook()

for participant_folder in participants_folders:
    print(participant_folder[58:])
    # To save only team name as an excel sheet instead of path of the folder containing predicted masks

    # add_sheet is used to create sheet of each participant
    ccbt = wb.add_sheet(participant_folder[58:])
    ccbt.write(0, 0, 'Patient ID')
    ccbt.write(0, 1, 'Epithelial')
    ccbt.write(0, 2, 'Lymphocyte')
    ccbt.write(0, 3, 'Neutrophil')
    ccbt.write(0, 4, 'Macrophage')

    for image_count, filei in enumerate(files):
        ccbt.write(image_count+1, 0, filei) # Add image name in excel file

        # Ambiguous_region which was provided with the testing data to exclude from the metric computation
        imgs = glob.glob(filei+'/**/**')
        ambiguous_idx = [i for i, f_name in enumerate(imgs) if 'Ambiguous' in f_name]

        # Check if abmiguous_idx exists
        if ambiguous_idx:
            ambiguous_regions = sio.loadmat(imgs[ambiguous_idx[0]])['n_ary_mask']
            ambiguous_regions = 1-(ambiguous_regions > 0)
            imgs.pop(ambiguous_idx[0])

        for i, f_name in enumerate(imgs):
            print(f_name)
            class_id = ([idx for idx in range(len(cell_types)) if cell_types[idx] in f_name])
            # Cell-type

            # Read ground truth image
            ground_truth = sio.loadmat(f_name)['n_ary_mask']

            # Read predicted mask and exclude the ambiguous regions for metric computation
            pred_img_name = glob.glob(participant_folder+'/'+filei+'/'+cell_types[class_id[0]]+'/**')

            if not pred_img_name:
                ccbt.write(image_count+1, class_id[0]+1, 0)
                print(0)
            else:
                predicted_mask = sio.loadmat(pred_img_name[0])

                mask_saved = ['img','name','n_ary_mask','Neutrophil_mask','arr','item','Epithelial_mask','Macrophage_mask', 'Lymphocyte_mask']
                mask_key = [m for m in mask_saved if m in predicted_mask.keys()]
                predicted_mask = predicted_mask[mask_key[0]]

                # Converting binary to n-ary mask for those participants who did not send masks as informed
                if len(np.unique(predicted_mask)) == 2:
                    predicted_mask, num_features = scipy.ndimage.measurements.label(predicted_mask)

                print(pred_img_name)

                if ambiguous_idx:
                    predicted_mask = predicted_mask*ambiguous_regions

                # Compute Panoptic Quality
                PQ = Panoptic_quality(ground_truth, predicted_mask)
                print(PQ)

                ccbt.write(image_count+1, class_id[0]+1, PQ)

wb.save('MoNuSAC-testing-PQ.xls')
