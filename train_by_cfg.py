import os
import logging
import time
import configparser
import argparse

import nibabel as nib
import numpy as np

from metric import *
from keras.optimizers import *
from keras import backend as K

from loss import dice_loss_multi_class_no_background
from keras.losses import categorical_crossentropy
from shutil import copyfile

from util import generatePartitionTrainAndValidFromFolderRandomly
from dataGenerator import DataGenerator
from keras.callbacks import *
from unet_3D import UNet_3D

def determine_crossentropy_weights_from_segmentation_mask(mp):
    masks = os.listdir(mp['labelPath'])
    
    weights = np.zeros(mp['labels'])
    S = 0
    img = nib.load(os.path.join(mp['labelPath'], masks[0]))
    seg = img.get_fdata().astype('uint8')    
    for idx in range(mp['labels']):
        weights[idx] = np.sum(seg==idx) 
        S += weights[idx]
    weights = S/weights
    weights = weights / np.sum(weights)
    return weights

def collect_parameters(cp, section):
    paramdict = {}
    
    for entry in cp[section]:
        if not any(i.isalpha() for i in cp[section][entry]):
            if '.' in cp[section][entry]:
                paramdict.update({entry: float(cp[section][entry])})
            else:
                paramdict.update({entry: int(cp[section][entry])})
        else:
            paramdict.update({entry: cp[section][entry]})
    
    return paramdict

def assemble_additional_parameters(mp, gen_param):
    params = {'dim': (mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['z_end']-mp['z_start']),
              'crop_parameters': [mp['x_start'], mp['x_end'], mp['y_start'], mp['y_end'], mp['z_start'], mp['z_end']],
              'batch_size': mp['batchsize'],
              'n_classes': mp['labels'],
              'n_channels': mp['channels'],
              'imageType': mp['imgType'],
              'labelType': mp['labelType'],
              'variableType': gen_param['variableType'],
              'loss_weights': [gen_param['lossWeightsLower'], gen_param['lossWeightsUpper']]}
    return params
    
def create_data_storage(mp, config, partition):
    
    model_path  = os.path.join(mp['savmodpath'] + time.strftime("%Y%m%d_%H%M%S") + '_'  + mp['comment'] + '/model.h5')
    
    save_path   = os.path.dirname(model_path)
    log_path    = save_path + '/logs'
    
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
        os.mkdir(log_path)
    
    current_dir = os.path.dirname(os.path.realpath("__file__"))
    copyfile(current_dir + '/' + config, save_path + '/'  + config)
    
    # Save the validation partition cases.
    f = open(save_path+'/validation_partition.txt',"w")
    for p in partition['validation']:
        f.write(p + '\n')
    f.close()
    
    return save_path, log_path, model_path

def create_callbacks(cb_param, log_path):
    callbacks = []
    
    if cb_param['earlyStop']:
        earlystopper = EarlyStopping(monitor='val_loss', patience=cb_param['earlyStopPatience'], verbose=1, mode='auto')
    
    callbacks.append(earlystopper)
    
    if cb_param['historyLog']:
        csv_log_cback = CSVLogger('{}/historyLog.csv'.format(log_path))
        
    callbacks.append(csv_log_cback)
    
    return callbacks

def create_loss_func(mp):
    if mp['loss'] == 'dice_multi_class_nb_plus_xentropy':
        def lossfunc(y_true, y_pred):
            return dice_loss_multi_class_no_background(y_true, y_pred, mp['labels']) + categorical_crossentropy(y_true, y_pred)
    return lossfunc
    if mp['loss'] == 'dice_multi_class_nb_plus_weighted_xentropy':
        weights = determine_weights_from_segmentation_mask(mp['labelPath'])
        def lossfunc(y_true, y_pred):
            weights = K.variable(weights)

            def weighted_categorical_crossentropy(y_true, y_pred):
                y_pred /= K.clip(y_pred, K.epsilon(), 1-K.epsilon())
                loss = y_true * K.log(y_pred) * weights
                loss = -K.sum(loss, -1)

            return dice_loss_multi_class_no_background(y_true, y_pred, mp['labels']) + weighted_categorical_crossentropy(y_true, y_pred) 

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
    parser.add_argument("-c", "--config", help="Config file for training function.", default="", type=str)
    parser.add_argument("-g", "--gpu", help="Select gpu (0,1,2,3,4 on giggs).", default='0', type=str)
    
    args = parser.parse_args()
    
    # Load config file.
    p = configparser.ConfigParser()
    p.optionxform = str
    p.read(args.config)
    
    # Collect main, data generation, normalization, image, augmentation and callback settings.
    mp         = collect_parameters(p, 'MAIN')
    gen_param  = collect_parameters(p, 'GEN')
    norm_param = collect_parameters(p, 'NORM')
    aug_param  = collect_parameters(p, 'AUG')
    cb_param   = collect_parameters(p, 'CALLBACK')
    
    misc_param = assemble_additional_parameters(mp, gen_param)
    
    # Set up graphics card settings.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Generate training/test partition.
    partition = generatePartitionTrainAndValidFromFolderRandomly(mp['imgPath'], gen_param['validprop'], gen_param['shuffletrain'], mp['imgType'])
    
    # Generate data generator objects.
    training_generator   = DataGenerator(partition['train'], mp['imgPath'], mp['labelPath'], norm_param, mp['augment'], aug_param, **misc_param)
    validation_generator = DataGenerator(partition['validation'], mp['imgPath'], mp['labelPath'], norm_param, mp['augmentval'], aug_param, **misc_param)
    
    # Generate the model.
    n_rows, n_cols, n_slices = mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['z_end']-mp['z_start']
    model = UNet_3D(input_shape=(n_rows, n_cols, n_slices, mp['channels']), nb_labels=mp['labels'], filters=mp['features'], depth=mp['depth'], nb_bottleneck=mp['bneck'], activation=mp['activation'], activation_network=mp['outact'], 
                    batch_norm=mp['batchnorm'], dropout_encoder=mp['dropout_en'], dropout_decoder=mp['dropout_de'], use_preact=False, use_mvn=False).create_model()
    model.summary()                
    # Set up loss function and metrics.
    metrics  = [dice_coef_background, dice_coef_first_label, dice_coef_second_label, dice_coef_third_label, dice_coef_fourth_label]
    lossfunc = create_loss_func(mp)
        
    # Compile model.
    #if mp['optimizer'] == 'adam':
    #    print('Adam optimizer')
    #    opt = Adam(lr=1e-3)
    #    model.compile(optimizer=opt, loss=lossfunc, metrics=metrics)
    #else:
    model.compile(optimizer=mp['optimizer'], loss=lossfunc, metrics=metrics)
    model.summary()
    # Print report prior to fit of the model.
    print_report(p)
    
    # Generate data storage folders.
    save_path, log_path, model_path = create_data_storage(mp, args.config, partition)
    
    # Generate callback settings.
    callbacks = create_callbacks(cb_param, log_path)
    
    # Fit model.
    results = model.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=True, epochs=mp['epochs'], callbacks=callbacks)
    
    # Store model.
    model.save(model_path)
