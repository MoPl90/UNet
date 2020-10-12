from keras import backend as K

import numpy as np

def dice_coef(y_true, y_pred, axis=(1, 2, 3), epsilon=1e-6):
    y_pred_cal = y_pred[:, :, :, :, 1]
    y_true_cal = y_true[:, :, :, :, 1]


    intersection = K.sum(y_true_cal * y_pred_cal, axis=axis)
    area_true = K.sum(y_true_cal, axis=axis)
    area_pred = K.sum(y_pred_cal, axis=axis)

    summation = area_true + area_pred
    return K.mean((2.0 * intersection + epsilon) / (summation + epsilon), axis=0)

def dice_coef_multi_class(y_true, y_pred, axis=(1, 2, 3), epsilon=1e-6):


    intersection = K.sum(y_true * y_pred, axis=axis)
    area_true = K.sum(y_true_ axis=axis)
    area_pred = K.sum(y_pred, axis=axis)

    summation = area_true + area_pred
    return K.mean((2.0 * intersection + epsilon) / (summation + epsilon), axis=0)
 
def jaccard_coef(y_true, y_pred, axis=(1, 2, 3), epsilon=1e-6):
    y_pred_cal = y_pred[:, :, :, :, 1]
    y_true_cal = y_true[:, :, :, :, 1]

    intersection = K.sum(y_true_cal * y_pred_cal, axis=axis)
    area_true = K.sum(y_true_cal, axis=axis)
    area_pred = K.sum(y_pred_cal, axis=axis)

    union = area_true + area_pred - intersection
    return K.mean((intersection + epsilon) / (union + epsilon), axis=0)

def euclidean_distance_loss_binary(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    return np.mean(np.sqrt(np.sum(np.square(result ^ reference), axis=-1)))   

# Dice coefficients for the VCGM scheme.

def dice_coef_background(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 0], y_pred[:, :, :,  :, 0])

def dice_coef_first_label(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 1], y_pred[:, :, :, :, 1])

def dice_coef_second_label(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 2], y_pred[:, :, :, :, 2])

def dice_coef_third_label(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 3], y_pred[:, :, :, :, 3])

def dice_coef_fourth_label(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 4], y_pred[:, :, :, :, 4])

# Dice coefficients for the QN labelling scheme.

def d_background(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 0], y_pred[:, :, :, :, 0])

def d_white_matter_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 1], y_pred[:, :, :, :, 1])

def d_grey_matter_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 2], y_pred[:, :, :, :, 2])

def d_white_matter_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 3], y_pred[:, :, :, :, 3])

def d_grey_matter_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 4], y_pred[:, :, :, :, 4])

def d_lateral_ventricle_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 5], y_pred[:, :, :, :, 5])

def d_cerebellar_white_matter_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 6], y_pred[:, :, :, :, 6])

def d_cerebellar_grey_matter_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 7], y_pred[:, :, :, :, 7])

def d_thalamus_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 8], y_pred[:, :, :, :, 8])

def d_caudate_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 9], y_pred[:, :, :, :, 9])

def d_putamen_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 10], y_pred[:, :, :, :, 10])

def d_pallidum_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 11], y_pred[:, :, :, :, 11])

def d_third_ventricle(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 12], y_pred[:, :, :, :, 12])

def d_fourth_ventricle(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 13], y_pred[:, :, :, :, 13])

def d_brain_stem(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 14], y_pred[:, :, :, :, 14])

def d_hippocampus_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 15], y_pred[:, :, :, :, 15])

def d_amygdala_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 16], y_pred[:, :, :, :, 16])

def d_ventral_dc_left(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 17], y_pred[:, :, :, :, 17])

def d_lateral_ventricle_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 18], y_pred[:, :, :, :, 18])

def d_cerebellar_white_matter_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 19], y_pred[:, :, :, :, 19])

def d_cerebellar_grey_matter_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 20], y_pred[:, :, :, :, 20])

def d_thalamus_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 21], y_pred[:, :, :, :, 21])

def d_caudate_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 22], y_pred[:, :, :, :, 22])

def d_putamen_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 23], y_pred[:, :, :, :, 23])

def d_pallidum_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 24], y_pred[:, :, :, :, 24])

def d_hippocampus_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 25], y_pred[:, :, :, :, 25])

def d_amygdala_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 26], y_pred[:, :, :, :, 26])

def d_ventral_dc_right(y_true, y_pred):
    return dice_coef_multi_class(y_true[:, :, :, :, 27], y_pred[:, :, :, :, 27])
