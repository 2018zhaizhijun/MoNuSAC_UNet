# coding=utf-8
import numpy as np
import cv2
import random


class ImageRotation:
    def __init__(self):
        self.img_rotated = None
        self.mask_rotated = None
        self.cnt_rotated = None

    def apply(self, img, mask, contour):
        rd = random.randint(0, 1)
        if rd == 0:
            return img, mask, contour
        angle = np.random.randint(0, 360)
        w, h = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        self.img_rotated = cv2.warpAffine(img, M, (w, h))
        self.mask_rotated = cv2.warpAffine(mask, M, (w, h))
        self.cnt_rotated = cv2.warpAffine(contour, M, (w, h))
        return self.img_rotated, self.mask_rotated, self.cnt_rotated


class ImageFlipping:
    def __init__(self, flip_x=1, flip_y=1, flip_o=1):
        self.img_flipped = None
        self.mask_flipped = None
        self.cnt_flipped = None

    def apply(self, img, mask, contour):
        rd = random.randint(0, 1)
        if rd == 0:
            return img, mask, contour
        id = np.random.randint(0, 3)
        if id == 0:
            self.img_flipped = cv2.flip(img, 0)
            self.mask_flipped = cv2.flip(mask, 0)
            self.cnt_flipped = cv2.flip(contour, 0)
        elif id == 1:
            self.img_flipped = cv2.flip(img, 1)
            self.mask_flipped = cv2.flip(mask, 1)
            self.cnt_flipped = cv2.flip(contour, 1)
        elif id == 2:
            self.img_flipped = cv2.flip(img, -1)
            self.mask_flipped = cv2.flip(mask, -1)
            self.cnt_flipped = cv2.flip(contour, -1)
        return self.img_flipped, self.mask_flipped, self.cnt_flipped


class ImageAffine:
    def __init__(self, alpha_affine=15):
        self.alpha_affine = alpha_affine
        self.img_trans = None
        self.mask_trans = None
        self.cnt_trans = None

    def apply(self, img, mask, contour):
        rd = random.randint(0, 1)
        if rd == 0:
            return img, mask, contour
        shape_size = img.shape[:2]
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size,
                            center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + np.random.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        self.img_trans = cv2.warpAffine(img, M, shape_size, borderMode=cv2.BORDER_REFLECT_101)
        self.mask_trans = cv2.warpAffine(mask, M, shape_size, borderMode=cv2.BORDER_REFLECT_101)
        self.cnt_trans = cv2.warpAffine(contour, M, shape_size, borderMode=cv2.BORDER_REFLECT_101)
        return self.img_trans, self.mask_trans, self.cnt_trans


class Gaussian_Noise:
    def __init__(self, noise_sigma=25):
        self.noise_sigma = noise_sigma
        self.img_trans = None
        self.mask_trans = None
        self.cnt_trans = None

    def apply(self, img, mask, contour):
        rd = random.randint(0, 1)
        if rd == 0:
            return img, mask, contour
        temp_image = img
        w, h, _ = temp_image.shape
        # 标准正态分布*noise_sigma
        noise = np.random.randn(w, h) * self.noise_sigma
        noisy_image = np.zeros(temp_image.shape, dtype='uint8')
        if len(temp_image.shape) == 2:
            noisy_image = temp_image + noise
        else:
            noisy_image[:, :, 0] = temp_image[:, :, 0]
            noisy_image[:, :, 1] = temp_image[:, :, 1]
            noisy_image[:, :, 2] = temp_image[:, :, 2]
        self.img_trans = noisy_image
        self.mask_trans = mask
        self.cnt_trans = contour
        return self.img_trans, self.mask_trans, self.cnt_trans


# if __name__ == '__main__':
#     data = cv2.imread(r'D:\MyData\MoNuSAC2020\CIA-Net\data\MoNuSeg\TRAIN\Image\1.png')  # BGR
#     cv2.imshow("img", data)
#     cv2.waitKey()
#     data = np.array(data)
#     mask = np.array(cv2.imread(r'D:\MyData\MoNuSAC2020\CIA-Net\data\MoNuSeg\TRAIN\Mask\1.png', 0))
#     contour = np.array(cv2.imread(r'D:\MyData\MoNuSAC2020\CIA-Net\data\MoNuSeg\TRAIN\Contour\1.png', 0))
#     augs = [ImageRotation(), ImageFlipping(), ImageAffine(), Gaussian_Noise()]
#     for aug in augs:
#         data, mask, contour = aug.apply(data, mask, contour)
#         cv2.imshow("img", mask)
#         cv2.waitKey()
