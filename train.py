import os
import logging
import time
import configparser
import argparse
import datetime

import nibabel as nib
import numpy as np

from metric import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K

from loss import *
from tensorflow.keras.losses import categorical_crossentropy
from shutil import copyfile

from util import generatePartitionTrainAndValidFromFolderRandomly
from data_generator import DataGenerator
from data_generator_multi_thread import DataGenerator_MT
from tensorflow.keras.callbacks import *
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
    if not 'resize_parameters' in mp:
        mp['resize_parameters'] = None

    params = {'dim': (mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['z_end']-mp['z_start']),
              'crop_parameters': [mp['x_start'], mp['x_end'], mp['y_start'], mp['y_end'], mp['z_start'], mp['z_end']],
              'resize_parameters': mp['resize_parameters'],
              'batch_size': mp['batchsize'],
              'n_classes': mp['labels'],
              'n_channels': mp['channels'],
              'imageType': mp['imgType'],
              'labelType': mp['labelType'],
              'variableTypeX': gen_param['variableTypeX'],
              'variableTypeY': gen_param['variableTypeY'],
              'loss_weights': [gen_param['lossWeightsLower'], gen_param['lossWeightsUpper']]}
    return params

def verify_parameters(mp):
    # Check if the dimension of the input volume.
    dims = mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['z_end']-mp['z_start']

    assert np.sum(np.mod(dims, 2**mp['depth']))==0, "All input dimensions have to be divisible by 2**Depth."


def create_data_storage(mp, config, partition, train, out_folder):

    if out_folder=='':
        model_path = os.path.join(mp['savmodpath'] + time.strftime("%Y%m%d_%H%M%S")  + '_' + 'Dropout' + str(train) + '_'  + mp['comment'] + '/model.h5')
    else:
        model_path = out_folder
    save_path   = os.path.dirname(model_path)
    log_path    = save_path + '/logs'
    script_path = save_path + '/scripts'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(log_path)
        os.mkdir(script_path)

    current_dir = os.path.dirname(os.path.realpath("__file__"))
    copyfile(current_dir + '/' + config, save_path + '/'  + config.split('/')[-1])

    # Save the validation partition cases.
    f = open(save_path+'/validation_partition.txt',"w")
    for p in partition['validation']:
        f.write(p + '\n')
    f.close()

    # Save the used train-/unet_3D- and operations.py files
    try:
        os.system("cp {} {}".format(sys.argv[0], os.path.join(script_path, sys.argv[0])))
        os.system("cp {} {}".format('unet_3D.py', os.path.join(script_path, 'unet_3D.py')))
        os.system("cp {} {}".format('operations.py', os.path.join(script_path, 'operations.py')))
    except:
        pass

    return save_path, log_path, model_path


def create_callbacks(cb_param, save_path, log_path):
    callbacks = []

    if cb_param['earlyStop']:
        earlystopper = EarlyStopping(monitor='val_loss', patience=cb_param['earlyStopPatience'], verbose=1, mode='auto')
        callbacks.append(earlystopper)

    if cb_param['historyLog']:
        csv_log_cback = CSVLogger('{}/historyLog.csv'.format(log_path))
        callbacks.append(csv_log_cback)

    modelcheck = ModelCheckpoint(filepath = os.path.join(save_path, 'model_{val_loss:.2f}_{epoch:02d}.h5'), save_best_only = True)
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
                y_true = K.cast(y_true, 'float32')
                y_pred = K.cast(y_pred, 'float32')

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


def run_training(args):
    if args.train=='1':
        train=True
    else:
        train=False

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

    # Verify parameters.
    verify_parameters(mp)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Generate training/test partition.
    if mp['trnImgPath'] == mp['valImgPath']:
        partition = generatePartitionTrainAndValidFromFolderRandomly(mp['trnImgPath'], gen_param['validprop'], gen_param['shuffletrain'], mp['imgType'], mp['trnLabelPath'], mp['labelType'], gen_param['trnThreshold'])
        print('No separate valImgPath stated -> splitted data into train and validation partition')
    else:
        partition = generatePartitionTrainAndValidFromFolderRandomly(mp['trnImgPath'], 0.0, gen_param['shuffletrain'], mp['imgType'], mp['trnLabelPath'], mp['labelType'], gen_param['trnThreshold'])
        partition['validation'] = generatePartitionTrainAndValidFromFolderRandomly(mp['valImgPath'], 0.0, gen_param['shuffletrain'], mp['imgType'], mp['valLabelPath'], mp['labelType'], gen_param['valThreshold'])['train']


    # Generate data storage folders.
    save_path, log_path, model_path = create_data_storage(mp, args.config, partition, train, args.out)
    misc_param['savePath'] = save_path

    # Generate data generator objects.
    if args.multi_threaded=='0':
        training_generator   = DataGenerator(partition['train'], mp['trnImgPath'], mp['trnLabelPath'], norm_param, mp['augment'], aug_param, **misc_param)
        validation_generator = DataGenerator(partition['validation'], mp['valImgPath'], mp['valLabelPath'], norm_param, mp['augmentval'], aug_param, **misc_param)
    elif args.multi_threaded=='1':
        training_generator   = DataGenerator_MT(mp['epochs'], partition['train'], mp['trnImgPath'], mp['trnLabelPath'], norm_param, mp['augment'], aug_param, **misc_param)
        validation_generator = DataGenerator_MT(mp['epochs'], partition['validation'], mp['valImgPath'], mp['valLabelPath'], norm_param, mp['augmentval'], aug_param, **misc_param)


    # Generate the model.
    n_rows, n_cols, n_slices = mp['x_end']-mp['x_start'], mp['y_end']-mp['y_start'], mp['z_end']-mp['z_start']

    model = UNet_3D(input_shape=(n_rows, n_cols, n_slices, mp['channels']), nb_labels=mp['labels'],
                    depth=mp['depth'], nb_bottleneck=mp['bneck'], filters=mp['features'], activation=mp['activation'], activation_network=mp['outact'],
                    dropout_encoder=mp['dropout_en'], dropout_decoder=mp['dropout_de'], train=train).create_model()

    # Set up  metrics.
    metrics = []
    for i in range(int(mp['labels'])):
        metrics.append(dice_coef_label(i))

    # Set up loss function
    lossfunc = create_loss_func(mp)

    # Set up optimizer
    if mp['optimizer'].lower() in ['radam', 'ranger']:
        tot_update_steps = mp['epochs'] * len(partition['train'])
        opt = RectifiedAdam(learning_rate = mp['lr'],
                            total_steps = tot_update_steps,
                            warmup_proportion = 0.1,
                            min_lr = mp['lr'])
        if mp['optimizer'].lower() == 'ranger':
            opt = Lookahead(opt, sync_period=6, slow_step_size=0.5)

    elif mp['optimizer'].lower() == 'nadam':
        opt = Nadam(learning_rate = mp['lr'])

    elif mp['optimizer'].lower() == 'adam':
        opt = Adam(learning_rate = mp['lr'])

    elif mp['optimizer'].lower() == 'sgd':
        opt = SGD(learning_rate = mp['lr'])

    else:
        print('Invalid optimizer selection "{}" (may be not implemented)'.format(mp['optimizer']))
        exit()

    # Compile model
    model.compile(optimizer=opt, loss=lossfunc, metrics=metrics)

    # Generate callback settings.
    callbacks = create_callbacks(cb_param, save_path, log_path)

    # Fit model (multiprocessing = False because of warnings)
    results = model.fit(x=training_generator, validation_data=validation_generator, use_multiprocessing=False, epochs=mp['epochs'], callbacks=callbacks, shuffle=False, verbose=1)

    # Store model.
    model.save(model_path)

if __name__ == "__main__":
    
    # Parse input arguments of the train method.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Config file for training function.", default="", type=str)
    parser.add_argument("-g", "--gpu", help="Select gpu (0,1,2,3,4 on giggs).", default='0', type=str)
    parser.add_argument("-t", "--train", help="Set dropout layers active during prediction, for MC estimates.", default='0', type=str) 
    parser.add_argument("-o", "--out", help="Output path for model etc..", default='', type=str)
    parser.add_argument("-m", "--multi_threaded", help="Multi-threaded data generator.", default='0')
    
    args = parser.parse_args()
    
    run_training(args)
