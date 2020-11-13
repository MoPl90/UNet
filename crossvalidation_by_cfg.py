import os
import logging
import time
import configparser
import argparse

from metric import *

from loss import dice_loss_multi_class_no_background
from keras.losses import categorical_crossentropy
from shutil import copyfile

from util import generatePartitionTrainAndValidFromFolderRandomly, generatePartitionCrossValidation
from dataGenerator import DataGenerator
from keras.callbacks import *
from unet_3D import UNet_3D

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

def initialize_data_storage_cross_validation(mp, config):
    
    base_path   = os.path.join(mp['savmodpath'] + '/Cross-Validation_' + time.strftime("%Y%m%d_%H%M%S") + '_'  + mp['comment'])
    
    os.mkdir(base_path)
    
    current_dir = os.path.dirname(os.path.realpath("__file__"))
    copyfile(current_dir + '/' + config, base_path + '/'  + config.split('/')[-1])
    
    return base_path

def create_model_storage(base_path, cv_index, partition):
    
    model_path  = os.path.join(base_path + '/' + str(cv_index) + '/model.h5')
    
    log_path    = os.path.dirname(model_path) + '/logs'
    
    if os.path.exists(model_path) is False:
        os.mkdir(os.path.dirname(model_path))
        os.mkdir(log_path)
    
    # Save the validation partition cases.
    f = open(os.path.dirname(model_path) + '/validation_partition.txt', "w")
    for p in partition['validation']:
        f.write(p + '\n')
    f.close()
    
    return log_path, model_path

def create_callbacks(cb_param, model_path, log_path):
    callbacks   = []
    
    # tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True)
    # callbacks.append(tensorboard)
    
    checkpoint  = ModelCheckpoint(filepath=os.path.join(os.path.dirname(model_path), 'weights.{epoch:04d}.hdf5'), verbose=0, save_best_only=False, mode='auto')
    callbacks.append(checkpoint)
    
    if cb_param['earlyStop']:
        earlystopper = EarlyStopping(monitor='val_loss', patience=cb_param['earlyStopPatience'], verbose=1, mode='auto')
    
    callbacks.append(earlystopper)
    
    if cb_param['historyLog']:
        csv_log_cback = CSVLogger('{}/historyLog.csv'.format(log_path))
        
    callbacks.append(csv_log_cback)
    
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

    if loss_name == 'XentropyWeighted':
        def lossfunc(y_true, y_pred):
            return categorical_xentropy_loss(y_true, y_pred, loss_weights)
        return lossfunc

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
    parser.add_argument("-c", "--config", help="Config file for training function.", default="", type=str)
    parser.add_argument("-k", "--chunks", help="Number of chunks.", default="2")
    parser.add_argument("-g", "--gpu", help="Select gpu (0,1,2,3,4 on giggs).", default="0", type=str)
    
    args = parser.parse_args()
    
    # Load config file.
    p = configparser.ConfigParser()
    p.optionxform = str
    p.read(args.config)
    
    # Collect main, data generation, normalization, image, augmentation and callback settings.
    mp         = collect_parameters(p, 'MAIN')
    gen_param  = collect_parameters(p, 'GEN')
    val_param  = collect_parameters(p, 'VAL')
    norm_param = collect_parameters(p, 'NORM')
    aug_param  = collect_parameters(p, 'AUG')
    cb_param   = collect_parameters(p, 'CALLBACK')
    
    misc_param = assemble_additional_parameters(mp, gen_param)
    
    # Set up graphics card settings.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Generate training/test partition.
    chunks = generatePartitionCrossValidation(imagePathFolder = gen_param['imgPath'], 
                                              k               = int(args.chunks), 
                                              shuffle_train   = gen_param['shuffletrain'], 
                                              labelPathFolder = gen_param['labelPath'],
                                              image_type      = gen_param['imgType'], 
                                              label_type      = gen_param['labelType'], 
                                              threshold       = gen_param['thresh'])
    
    # Generate basic data storage folder.
    base_path = initialize_data_storage_cross_validation(mp, args.config)

    for cv_index in range(len(chunks)):
        trainset = []
        for i in range(len(chunks)):
            if not i==cv_index:
                trainset = trainset + chunks[i]
        valset = chunks[cv_index]
        
        partition = dict()
        partition['train'] = trainset
        partition['validation']   = valset
        
        # Generate data generator objects.
        training_generator   = DataGenerator(partition['train'], gen_param['imgPath'], gen_param['labelPath'], norm_param, gen_param['augment'], aug_param, **misc_param)
        validation_generator = DataGenerator(partition['validation'], gen_param['imgPath'], gen_param['labelPath'], norm_param, gen_param['augment'], aug_param, **misc_param)
    
        # Generate the model.
        n_rows, n_cols, n_slices = mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['z_end']-mp['z_start']
        model = UNet_3D(input_shape=(n_rows, n_cols, n_slices, mp['channels']), nb_labels=mp['labels'], filters=mp['features'], depth=mp['depth'], nb_bottleneck=mp['bneck'], activation=mp['activation'], activation_network=mp['outact'], 
                        batch_norm=mp['batchnorm'], dropout_encoder=mp['dropout_en'], dropout_decoder=mp['dropout_de'], use_preact=False, use_mvn=False).create_model()
                    
        # Set up loss function and metrics.
        metrics  = [dice_coef_background, dice_coef_first_label]
        lossfunc = create_loss_func(mp)
        
        # Compile model.
        model.compile(optimizer=mp['optimizer'], loss=lossfunc, metrics=metrics)

        # Print report prior to fit of the model.
        print_report(p)
        
        log_path, model_path = create_model_storage(base_path, cv_index, partition)
    
        # Generate callback settings.
        callbacks = create_callbacks(cb_param, model_path, log_path)
    
        # Fit model.
        results = model.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=False, epochs=mp['epochs'], callbacks=callbacks)
    
        # Store model.
        model.save(model_path)
