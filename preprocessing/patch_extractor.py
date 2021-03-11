import cv2


class PatchExtractor:
    def __init__(self, patch_size=(316, 316), patch_step=(194, 194)):
        self.patch_size = patch_size
        self.patch_step = patch_step
        self.img_patches = []
        self.mask_patches = []
        self.cnt_patches = []
        self.win_x = 0
        self.win_y = 0

    def extract(self, img_in, mask_in, contour_in):
        '''
        :param img: [w, h, channel]
        :param mask: [w, h]
        :return:
        '''
        img = cv2.copyMakeBorder(img_in, 46, 46, 46, 46, cv2.BORDER_REFLECT)
        mask = cv2.copyMakeBorder(mask_in, 46, 46, 46, 46, cv2.BORDER_REFLECT)
        contour = cv2.copyMakeBorder(contour_in, 46, 46, 46, 46, cv2.BORDER_REFLECT)
        mx, my = img.shape[:2]
        # assert mx == mask.shape[0] and my == mask.shape[1], "The size of masks must be the same as images."
        while self.win_y + self.patch_size[1] <= my:
            while self.win_x + self.patch_size[0] <= mx:
                self.img_patches.append(img[self.win_x:self.win_x+self.patch_size[0],
                                     self.win_y:self.win_y+self.patch_size[1]])
                self.mask_patches.append(mask[self.win_x+46:self.win_x+self.patch_size[0]-46,
                                     self.win_y+46:self.win_y+self.patch_size[1]-46])
                self.cnt_patches.append(contour[self.win_x+46:self.win_x+self.patch_size[0]-46,
                                         self.win_y+46:self.win_y+self.patch_size[1]-46])
                self.win_x += self.patch_step[0]
            self.win_x = 0
            self.win_y += self.patch_step[1]
        return self.img_patches, self.mask_patches, self.cnt_patches
