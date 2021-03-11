from model import UNet, SUNet, DUNet
import cv2
import numpy as np
from config import BGR_mean
import matplotlib.pyplot as plt


def predict(model, im):
    im = np.array(im).astype('float64')
    im = (im - np.array(BGR_mean, dtype='float64')) / 255.0
    im = np.expand_dims(im, 0)
    prediction = model.predict(im)
    prediction[prediction >= 0.5] = 255
    prediction[prediction < 0.5] = 128
    pred_mask = prediction.astype('uint8').squeeze(0).reshape(224, 224)
    pred_mask[contour == 255] = 0
    return pred_mask


model_unet = UNet()
model_unet.load_weights('Checkpoints/ep045-loss0.186-val_loss0.192.hdf5')

model_sunet = SUNet()
model_sunet.load_weights('SCheckpoints/ep048-loss0.201-val_loss0.193.hdf5')

model_dunet = DUNet()
model_dunet.load_weights('DCheckpoints/ep024-loss0.202-val_loss0.193.hdf5')

im = cv2.imread(r'data\MoNuSeg\VAL\Image\36.png')
im_pad = cv2.imread(r'data\MoNuSeg\VAL_UN\Image\122.png')
contour = cv2.imread(r'data\MoNuSeg\VAL_UN\Contour\122.png', cv2.IMREAD_GRAYSCALE)

pred_mask_sunet = predict(model_sunet, im)
pred_mask_unet = predict(model_unet, im_pad)
pred_mask_dunet = predict(model_dunet, im)

# pred_mask = Image.fromarray(pred_mask)

plt.subplot(2, 2, 1)
plt.imshow(im)
plt.subplot(2, 2, 2)
plt.imshow(pred_mask_unet, cmap="gray")
plt.subplot(2, 2, 3)
plt.imshow(pred_mask_sunet, cmap="gray")
plt.subplot(2, 2, 4)
plt.imshow(pred_mask_dunet, cmap="gray")
plt.show()
