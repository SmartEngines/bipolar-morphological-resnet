from __future__ import print_function
import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import os
from utils import *
from resnet import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Training parameters
batch_size = 1024
epochs = 600
data_augmentation = True
num_classes = 10
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 2

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 2

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = resnet_v2(input_shape=input_shape, depth=depth)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'cifar_saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
earlystop = EarlyStopping()
callbacks = [checkpoint, lr_reducer, lr_scheduler, earlystop]

datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None)

#Run training of standard model
datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                 validation_data=(x_test, y_test),
                                 epochs=epochs, verbose=1, workers=4,
                                 callbacks=callbacks, steps_per_epoch=x_train.shape[0] / batch_size)

# Score trained model.
model = resnet_v2(input_shape=input_shape, depth=depth)
best_path = os.path.join(save_dir, sorted(os.listdir(save_dir))[-1])
model.load_weights(best_path)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss of base model:', scores[0])
print('Test accuracy of base model:', scores[1])


analyze = Analyze(x_train, y_train)

start = 1
for i in range(1, 23):
    new_model = resnet_v2(input_shape=input_shape, depth=depth, morph_num=range(0, i))
    init_weights_from_model(new_model, model)
    new_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
    new_model.summary()
    scores = new_model.evaluate(x_test, y_test, verbose=1)
    print('Test loss after conversion:', scores[0])
    print('Test accuracy after conversion:', scores[1])

    save_dir = os.path.join(os.getcwd(), 'cifar_saved_models_bm'+str(i-1))
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
    callbacks = [checkpoint, lr_reducer, lr_scheduler, analyze]
   
    if i < start:
        ep = 0
    else:
        ep = 50
        batch_size = 32

   
    new_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=ep, verbose=1, workers=4,
                            callbacks=callbacks, steps_per_epoch=x_train.shape[0] / (batch_size ))
    model = new_model
    best_path = os.path.join(save_dir, sorted(os.listdir(save_dir))[-1])
    print('Loading weights from ', best_path)
    model.load_weights(best_path)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss after additional training:', scores[0])
    print('Test accuracy after additional training:', scores[1])

model = resnet_v2(input_shape=input_shape, depth=depth, morph_num=range(0, 22))
save_dir = os.path.join(os.getcwd(), 'cifar_saved_models_bm21')
best_path = os.path.join(save_dir, sorted(os.listdir(save_dir))[-1])
model.load_weights(best_path)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0) * 0.5),
              metrics=['accuracy'])
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


save_dir = os.path.join(os.getcwd(), 'cifar_saved_models_final')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
    # Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
callbacks = [checkpoint, lr_reducer, lr_scheduler, earlystop, analyze]
batch_size = 8
model.fit(x_train, y_train,
          validation_data=(x_test, y_test), batch_size=batch_size,
          epochs=epochs, verbose=1,
          callbacks=callbacks)

best_path = os.path.join(save_dir, sorted(os.listdir(save_dir))[-1])
print('Loading weights from ', best_path)
model.load_weights(best_path)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])