from keras import backend as K

from metric import *

# https://github.com/killthekitten/kaggle-carvana-2017/blob/master/losses.py
# https://github.com/chuckyee/cardiac-segmentation/blob/master/rvseg/loss.py
# https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/blob/master/zf_unet_224_model.py
# https://github.com/keras-team/keras/issues/3611
# https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution/blob/master/losses.py

# def dice_loss(y_true, y_pred, weights):
#     # Input tensors have shape (batch_size, height, width, classes)
#     # User must input list of weights with length equal to number of classes
#     #
#     # Ex: for simple binary classification, with the 0th mask
#     # corresponding to the background and the 1st mask corresponding
#     # to the object of interest, we set weights = [0, 1]
#     batch_dice_coefs = dice_coef(y_true, y_pred, axis=[1, 2])
#     dice_coefs = K.mean(batch_dice_coefs, axis=0)
#     w = K.constant(weights) / sum(weights)
#     return 1.0 - K.sum(w * dice_coefs)
#
#
# def jaccard_loss(y_true, y_pred, weights):
#     batch_jaccard_coefs = jaccard_coef(y_true, y_pred, axis=[1, 2])
#     jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
#     w = K.constant(weights) / sum(weights)
#     return 1.0 - K.sum(w * jaccard_coefs)

def dice_loss(y_true, y_pred, weights=[0.1, 0.9]):
    batch_coefs = dice_coef(y_true=y_true, y_pred=y_pred)
    w = K.constant(weights) / sum(weights)
    return 1.0 - K.sum(w * batch_coefs)

def dice_loss_multi_class(y_true, y_pred, numClasses=3):
    dice_multi=0
    for index in range(numClasses):
        dice_multi -= dice_coef_multi_class(y_true[:, :, :, :, index], y_pred[:, :, :, :, index])
    return dice_multi

def dice_loss_multi_class_no_background(y_true, y_pred, numClasses=3, epsilon=1e-6):
    dice_multi=0
    for index in range(numClasses-1):
        dice_multi -= dice_coef_multi_class(y_true[:, :, :, :, index+1], y_pred[:, :, :, :, index+1], epsilon=epsilon)
    return dice_multi

def jaccard_loss(y_true, y_pred, weights=[0.1, 0.9]):
    batch_coefs = jaccard_coef(y_true=y_true, y_pred=y_pred)
    w = K.constant(weights) / sum(weights)
    return 1.0 - K.sum(w * batch_coefs)

def categorical_xentropy_loss(y_true, y_pred, weights=[0.1, 0.9], epsilon=1e-8):
    print('categorical xentropy loss')
    print(weights)
    ndim = K.ndim(y_pred)
    ncategory = K.int_shape(y_pred)[-1]
    # scale predictions so class probabilities of each pixel sum to 1
    y_pred /= K.sum(y_pred, axis=(ndim - 1), keepdims=True)
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    w = K.constant(weights) * (ncategory / sum(weights))

    # first, average over all axis except classes
    cross_entropies = -K.mean(y_true * K.log(y_pred), axis=tuple(range(ndim - 1)))
    return K.sum(w * cross_entropies)

def euclidean_distance_loss(y_true, y_pred):
    y_pred_cal = y_pred[:, :, :, :, 1]
    y_true_cal = y_true[:, :, :, :, 1]

    return K.sqrt(K.sum(K.square(y_pred_cal - y_true_cal), axis=-1))

def certain_number_vixels_loss(y_true, y_pred):
    y_pred_cal = y_pred[:, :, :, :, 1]
    y_true_cal = y_true[:, :, :, :, 1]

    y_true_vixels = K.sum(y_pred_cal)
    y_pred_vixels = K.sum(y_true_cal)

    return K.abs(y_true_vixels - y_pred_vixels)
