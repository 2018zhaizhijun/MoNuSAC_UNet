from model import UNet, SUNet, DUNet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from preprocessing.data_aug import *
from data_generator import train_gen, val_gen
from config import PAD


train_dir = ''
val_dir = ''
augs = []
if PAD:
    train_dir = r'D:/MyData/MoNuSAC2020/CIA-Net/data/MoNuSeg/TRAIN_UN'
    val_dir = r'D:/MyData/MoNuSAC2020/CIA-Net/data/MoNuSeg/VAL_UN'
    augs = [ImageFlipping(), Gaussian_Noise()]
else:
    train_dir = r'D:/MyData/MoNuSAC2020/CIA-Net/data/MoNuSeg/TRAIN'
    val_dir = r'D:/MyData/MoNuSAC2020/CIA-Net/data/MoNuSeg/VAL'
    augs = [ImageRotation(), ImageFlipping(), ImageAffine(), Gaussian_Noise()]

train_generator = train_gen(train_dir, batch_size=1, augs=augs)
val_generator = val_gen(val_dir, batch_size=1)

model = None
checkpoint = None
if PAD:
    model = UNet()
    checkpoint = ModelCheckpoint('Checkpoints/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.hdf5',
                             monitor='val_loss', verbose=1, save_best_only=True, period=3)
else:
    model = DUNet()
    checkpoint = ModelCheckpoint('DCheckpoints/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.hdf5',
                                 monitor='val_loss', verbose=1, save_best_only=True, period=3)

learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1)
model.fit_generator(train_generator,
                    steps_per_epoch=600,
                    validation_data=val_generator,
                    validation_steps=150,
                    epochs=51,
                    initial_epoch=0,
                    callbacks=[checkpoint, learning_rate_scheduler])
