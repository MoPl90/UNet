import os
import logging
import time
import configparser
import argparse
import warnings
import datetime

import nibabel as nib
import numpy as np

from metric import *
from keras.optimizers import *
from keras import backend as K
from keras.models import load_model

from loss import *
from keras.losses import categorical_crossentropy
from shutil import copyfile

from util import generatePartitionTrainAndValidFromFolderRandomly
from dataGenerator import DataGenerator
from keras.callbacks import *
from unet_3D import UNet_3D

def determine_weights_from_segmentation_mask(mp):
    masks = os.listdir(mp['labelPath'])
    
    weights = np.zeros(mp['labels'])
    S = 0
    img = nib.load(os.path.join(mp['labelPath'], masks[10]))
    seg = img.get_fdata().astype('uint8')    
    for idx in range(mp['labels']):
        weights[idx] = np.sum(seg==idx) 
        S += weights[idx]
    weights = S/weights
    weights = weights / np.sum(weights)
    print(weights)
    return weights

def collect_parameters(cp, section):
    paramdict = {}
    
    for entry in cp[section]:
        if not any(i.isalpha() for i in cp[section][entry]):
            if '\n' in cp[section][entry]:
                paramdict.update({entry: [float(w) for w in cp[section][entry].split()]})
            elif '.' in cp[section][entry]:
                paramdict.update({entry: float(cp[section][entry])})
            else:
                paramdict.update({entry: int(cp[section][entry])})
        else:
            paramdict.update({entry: cp[section][entry]})
    
    return paramdict


#Make this more general
def assemble_additional_parameters(mp, gen_param):
    params = {'dim': (mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['z_end']-mp['z_start']),
              'crop_parameters': [mp['x_start'], mp['x_end'], mp['y_start'], mp['y_end'], mp['z_start'], mp['z_end']],
              'batch_size': mp['batchsize'],
              'n_classes': mp['labels'],
              'n_channels': mp['channels'],
              'imageType': gen_param['imgType'],
              'labelType': gen_param['labelType'],
              'variableType': gen_param['variableType'],
              'loss_weights': [gen_param['lossWeightsLower'], gen_param['lossWeightsUpper']]}
    return params
    
def create_data_storage(mp, config):
    
    model_path  = os.path.join(mp['savmodpath'] + mp['comment'] + '/model.h5')

    save_path   = os.path.dirname(model_path)
    log_path    = save_path + '/logs'
    
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
        os.mkdir(log_path)
    
    return save_path, log_path, model_path

def create_callbacks(cb_param, log_path):
    callbacks = []
    
    if cb_param['earlyStop']:
        earlystopper = EarlyStopping(monitor='val_loss', patience=cb_param['earlyStopPatience'], verbose=1, mode='auto')

    callbacks.append(earlystopper)

    if cb_param['historyLog']:
        csv_log_cback = CSVLogger('{}/historyLog.csv'.format(log_path))

    callbacks.append(csv_log_cback)

    modelcheck = ModelCheckpoint(filepath=model_path + 'model.{epoch:02d}-{val_loss:.2f}.h5')
    callbacks.append(modelcheck)
    
    # logdir = log_path + "scalars/"
    # tensorboard_callback = TensorBoard(log_dir=logdir)
    # print('Tensorboard Dir: ' + logdir)
    # callbacks.append(tensorboard_callback)

    return callbacks

def create_loss_func(mp):
    loss_name = mp['loss']
    loss_weights = mp["lossWeights"]
    #loss_weights = determine_weights_from_segmentation_mask(mp)

    if loss_name == 'pixel':
        def lossfunc(y_true, y_pred):
            return categorical_xentropy_loss(
                y_true, y_pred, loss_weights)

        return lossfunc
    elif loss_name == 'dice':
        def lossfunc(y_true, y_pred):
            return dice_loss(y_true, y_pred, loss_weights)

        return lossfunc
    elif loss_name == 'dice_multi_class':
        def lossfunc(y_true, y_pred):
            return dice_loss_multi_class(y_true, y_pred, args.n_classes)
        return lossfunc

    elif loss_name == 'dice_multi_class_nb':
        def lossfunc(y_true, y_pred):
            return dice_loss_multi_class_no_background(y_true, y_pred, args.n_classes)
        return lossfunc

    if loss_name == 'dice_multi_class_nb_plus_xentropy':
        def lossfunc(y_true, y_pred):
            return dice_loss_multi_class_no_background(y_true, y_pred, mp['labels']) + categorical_crossentropy(
                y_true, y_pred)
        return lossfunc

    elif loss_name == 'dicePlusXentropy':
        def lossfunc(y_true, y_pred):
            return dice_loss(y_true, y_pred, loss_weights) + categorical_xentropy_loss(
                y_true, y_pred, loss_weights)
        return lossfunc

    if loss_name == 'dicePlusXentropyWeighted':
        def lossfunc(y_true, y_pred):
            return 0.00008 * dice_loss(y_true, y_pred, loss_weights) + 0.99992 * categorical_xentropy_loss(
                y_true, y_pred, loss_weights)
        return lossfunc

    if loss_name == 'jaccard':
        def lossfunc(y_true, y_pred):
            return jaccard_loss(y_true, y_pred, loss_weights)
        return lossfunc

    if loss_name == 'euclideanDistance':
        def lossfunc(y_true, y_pred):
            return euclidean_distance_loss(y_true, y_pred)
        return lossfunc

    if loss_name == 'euclideanDistancePlusXentropyWeighted':
        def lossfunc(y_true, y_pred):
            return 0.0001 * euclidean_distance_loss(y_true, y_pred) + 0.9999 * categorical_xentropy_loss(y_true, y_pred,
                                                                                                         loss_weights)
        return lossfunc

    if loss_name == 'dice_multi_class_nb_plus_weighted_xentropy':
        def lossfunc(y_true, y_pred, weights):

            def weighted_categorical_crossentropy(y_true, y_pred, weights=weights):
                weights = K.variable(weights)
                y_pred /= K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
                loss = y_true * K.log(y_pred + K.epsilon()) * weights
                loss = -K.sum(loss, -1)
                return loss

            return dice_loss_multi_class_no_background(y_true, y_pred,
                                                       mp['labels']) + weighted_categorical_crossentropy(y_true, y_pred,
                                                                                                         weights=weights)

        return lambda y_true, y_pred: lossfunc(y_true, y_pred, weights=loss_weights)

    raise Exception("Unknown loss ({})".format(loss_name))


def print_report(cp):
    print('Training U-Net 3D'.ljust(50, '-').rjust(80, '-'))
    print(''.rjust(80, '-'))
    for sec in p:
        if not sec=='DEFAULT':
            print(sec)
            for e in p[sec]:
                print((e + ': ' + p[sec][e]))
    print(''.rjust(80, '-'))

if __name__ == "__main__":
    
    # Parse input arguments of the train method.
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to pre-trained model.", default="", type=str)
    parser.add_argument("-g", "--gpu", help="Select gpu (0,1,2,3,4 on giggs).", default='0', type=str)
    parser.add_argument("-e", "--epochs", help="Number of fine-tuning epochs.", default=10, type=int)
    parser.add_argument("-n", "--n_samples", help="Number of samples to train on.", default='0', type=int)
    args = parser.parse_args()

    # Load config file.
    p = configparser.ConfigParser()
    p.optionxform = str
    files = os.listdir(args.path)
    for f in files:
        if f.endswith(".cfg"):
            config = args.path + f
    p.read(config)
    
    # Collect main, data generation, normalization, image, augmentation and callback settings.
    mp         = collect_parameters(p, 'MAIN')
    gen_param  = collect_parameters(p, 'GEN')
    val_param  = collect_parameters(p, 'VAL')
    norm_param = collect_parameters(p, 'NORM')
    aug_param  = collect_parameters(p, 'AUG')
    cb_param   = collect_parameters(p, 'CALLBACK')

    mp["epochs"] = args.epochs
    mp["savmodpath"] = args.path + "/FINE_TUNING"
    mp["comment"] = "_ON_" + str(args.n_samples) + "_SAMPLES_" + str(args.epochs) + "_EPOCHS"
    gen_param["imgPath"] = "/scratch/mplatscher/fake_data/clinical/ims"
    gen_param["labelPath"] = "/scratch/mplatscher/fake_data/clinical/gts"
    gen_param["imgType"] = ".nii.gz"
    gen_param["labelType"] = ".nii"
    gen_param["suffletrain"] = True

    misc_param_gen = assemble_additional_parameters(mp, gen_param)
    misc_param_val = assemble_additional_parameters(mp, val_param)

    # Set up graphics card settings.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Generate training/test partition.

    #Training data is now clinical
    train_ids = generatePartitionTrainAndValidFromFolderRandomly(imagePathFolder = gen_param['imgPath'],
                                                                 _validProportion = gen_param['validprop'],
                                                                 shuffle_train_val = gen_param['shuffletrain'],
                                                                 threshold=20,
                                                                 labelPathFolder = gen_param['labelPath'],
                                                                 image_type = gen_param['imgType'],
                                                                 label_type = gen_param["labelType"])

    train_ids["train"] = train_ids["train"][:args.n_samples]

    # Test set remains the same
    test_ids = generatePartitionTrainAndValidFromFolderRandomly(imagePathFolder = val_param['imgPath'],
                                                                _validProportion = val_param['validprop'],
                                                                shuffle_train_val = val_param['shuffletrain'],
                                                                threshold=20,
                                                                labelPathFolder = val_param['labelPath'],
                                                                image_type = val_param['imgType'],
                                                                label_type = val_param["labelType"])

    # Generate data generator objects.
    training_generator   = DataGenerator(train_ids['train'], gen_param['imgPath'], gen_param['labelPath'], norm_param, gen_param["augment"], aug_param, **misc_param_gen)
    validation_generator = DataGenerator(test_ids['validation'], val_param['imgPath'], val_param['labelPath'], norm_param, val_param["augment"], aug_param, **misc_param_val)

    # Generate the model.
    model = load_model(args.path + "model.h5", compile=False)
    model.summary()

    # Set up  metrics.
    metrics = [dice_coef_background, dice_coef_first_label]
    # Set up loss function
    lossfunc = create_loss_func(mp)
    opt = Adam(lr=mp["learningrate"])
    model.compile(optimizer=opt, loss=lossfunc, metrics=metrics)

    # Print report prior to fit of the model.
    print_report(p)

    # Generate data storage folders.
    save_path, log_path, model_path = create_data_storage(mp, config)

    # Generate callback settings.
    callbacks = create_callbacks(cb_param, log_path)

    # Fit model.
    results = model.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=False, epochs=mp['epochs'], callbacks=callbacks)

    # Store model.
    model.save(model_path)
