import keras
from keras import backend as K
import tensorflow as tf
from BipolarMorphologicalConv2D import BipolarMorphologicalConv2D


def init_weights_from_model(bm_model, model):
    """Initializing BM-model from existing model

    An existing model bm_model is initialized with weigths from model,
    weights are copied for the same layers and approximated for conv (in model)
    and BM layers (in bm_model)

    # Arguments
       bm_model - model to obtain weights for
       model - model to take weights from

    # Returns
       Initialized BM-model
    """
    n_layers = len(bm_model.layers)
    for i in range(0, n_layers):
        model_weights = model.layers[i].get_weights()
        model_layer_name = model.layers[i].name
        bm_model_layer_name = bm_model.layers[i].name

        if model_weights:
            if model_layer_name.startswith('conv2d') and bm_model_layer_name.startswith('bipolar_morphological_conv2d'):
                weights_shift = 0.000001
                input_shift = 0.000001
                kernel1_weights = K.maximum(K.log(K.maximum(+model_weights[0], weights_shift)), tf.float32.min)
                kernel2_weights = K.maximum(K.log(K.maximum(-model_weights[0], weights_shift)), tf.float32.min)
                with tf.Session():
                    kernel1_weights = kernel1_weights.eval()
                    kernel2_weights = kernel2_weights.eval()
                bm_model.layers[i].set_weights([kernel1_weights, kernel2_weights, model_weights[1]])
            else:
                bm_model.layers[i].set_weights(model_weights)
    return bm_model


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


class Analyze(keras.callbacks.Callback):
    """
    A callback class for model analyzing
    """
    def __init__(self, x, y_true):
        self.x = x
        self.y_true = y_true
        super(Analyze, self).__init__()

    def on_train_begin(self, logs=None):
        pass



    def on_epoch_end(self, epoch, logs=None):
        pass


