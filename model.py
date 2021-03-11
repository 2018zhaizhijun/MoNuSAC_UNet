from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, Dropout, UpSampling2D, Cropping2D
from keras.optimizers import Adam


def UNet(pretrained_weights=None, input_shape=(316, 316, 3)):
    inputs = Input(input_shape)

    conv_e1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(inputs)
    conv_e1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv_e1)
    pool_e1 = MaxPooling2D(pool_size=(2, 2))(conv_e1)
    conv_e2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool_e1)
    conv_e2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv_e2)
    pool_e2 = MaxPooling2D(pool_size=(2, 2))(conv_e2)
    conv_e3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool_e2)
    conv_e3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv_e3)
    drop_e3 = Dropout(0.5)(conv_e3)
    pool_e3 = MaxPooling2D(pool_size=(2, 2))(drop_e3)
    conv_e4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool_e3)
    conv_e4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv_e4)
    drop_e4 = Dropout(0.5)(conv_e4)

    up_d3 = Conv2D(256, 1, activation='relu', padding='valid', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop_e4))
    crop_d3 = Cropping2D(((4,4), (4,4)))(conv_e3)
    merge_d3 = Concatenate(axis=3)([crop_d3, up_d3])
    conv_d3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge_d3)
    conv_d3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv_d3)

    up_d2 = Conv2D(128, 1, activation='relu', padding='valid', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv_d3))
    crop_d2 = Cropping2D(((16,16), (16,16)))(conv_e2)
    merge_d2 = Concatenate(axis=3)([crop_d2, up_d2])
    conv_d2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge_d2)
    conv_d2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv_d2)
    conv_d2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv_d2)

    up_d1 = Conv2D(64, 1, activation='relu', padding='valid', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv_d2))
    crop_d1 = Cropping2D(((42,42), (42,42)))(conv_e1)
    merge_d1 = Concatenate(axis=3)([crop_d1, up_d1])
    conv_d1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge_d1)
    conv_d1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv_d1)
    conv_out = Conv2D(1, 1, activation='sigmoid')(conv_d1)

    model = Model(input=inputs, output=conv_out)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# UNet without image padding and Crop layer
def SUNet(pretrained_weights=None, input_shape=(224, 224, 3)):
    inputs = Input(input_shape)

    conv_e1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv_e1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_e1)
    pool_e1 = MaxPooling2D(pool_size=(2, 2))(conv_e1)
    conv_e2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool_e1)
    conv_e2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_e2)
    pool_e2 = MaxPooling2D(pool_size=(2, 2))(conv_e2)
    conv_e3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool_e2)
    conv_e3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_e3)
    drop_e3 = Dropout(0.5)(conv_e3)
    pool_e3 = MaxPooling2D(pool_size=(2, 2))(drop_e3)
    conv_e4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool_e3)
    conv_e4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_e4)
    drop_e4 = Dropout(0.5)(conv_e4)

    up_d3 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop_e4))
    merge_d3 = Concatenate(axis=3)([conv_e3, up_d3])
    conv_d3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_d3)
    conv_d3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_d3)

    up_d2 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv_d3))
    merge_d2 = Concatenate(axis=3)([conv_e2, up_d2])
    conv_d2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_d2)
    conv_d2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_d2)
    conv_d2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_d2)

    up_d1 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv_d2))
    merge_d1 = Concatenate(axis=3)([conv_e1, up_d1])
    conv_d1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_d1)
    conv_d1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_d1)
    conv_out = Conv2D(1, 1, activation='sigmoid')(conv_d1)

    model = Model(input=inputs, output=conv_out)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# 加上先upsample再pool的路径的UNet
def DUNet(pretrained_weights=None, input_shape=(224, 224, 3)):
    inputs = Input(input_shape)

    conv_e1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv_e1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_e1)

    up_d1 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv_e1))
    conv_d1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up_d1)
    conv_d1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_d1)
    pool_d1 = MaxPooling2D(pool_size=(2, 2))(conv_d1)

    pool_e1 = MaxPooling2D(pool_size=(2, 2))(conv_e1)
    conv_e2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool_e1)
    conv_e2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_e2)
    drop_e2 = Dropout(0.5)(conv_e2)
    pool_e2 = MaxPooling2D(pool_size=(2, 2))(drop_e2)
    conv_e3 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool_e2)
    conv_e3 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_e3)
    drop_e3 = Dropout(0.5)(conv_e3)

    up_d3 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop_e3))
    merge_d3 = Concatenate(axis=3)([conv_e2, up_d3])
    conv_d3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_d3)
    conv_d3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_d3)

    up_d2 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv_d3))
    merge_d2 = Concatenate(axis=3)([conv_e1, up_d2])
    conv_d2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_d2)
    conv_d2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_d2)
    merge_d0 = Concatenate(axis=3)([conv_d2, pool_d1])
    conv_d0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge_d0)
    conv_d0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_d0)
    conv_out = Conv2D(1, 1, activation='sigmoid')(conv_d0)

    model = Model(input=inputs, output=conv_out)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
