# from keras import backend as K
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))

from tensorflow.keras.losses import categorical_crossentropy
from dataGenerator import DataGenerator
from unet_3D import UNet_3D
from keras.utils import plot_model
from util import generatePartitionTrainAndValidFromFolderRandomly, plot_learning_curve
from loss import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import logging
import time
from opts import *
from keras.callbacks import *
from shutil import copyfile
# from predict_3D import predict_3D
from keras.utils import  multi_gpu_model
from medpy.metric.binary import dc

############################################################################
# Parameters and gpu
############################################################################

args = parse_arguments()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.g

nGPU = args.nGPU

############################################################################
# Creating output directory
############################################################################

current_dir = os.path.dirname(os.path.realpath("__file__"))
# save_path = os.path.join(current_dir, 'models/' + time.strftime("%Y%m%d_%H%M%S") + args.comment + '/model.h5')
save_path = os.path.join(args.saveModelPath + time.strftime("%Y%m%d_%H%M%S") + args.comment + '/model.h5')

print(save_path)

log_path = os.path.dirname(save_path) + '/logs'

if os.path.exists(save_path) is False:
    os.makedirs(os.path.dirname(save_path))
    os.makedirs(log_path)

#copy opts.py in log folder
copyfile(current_dir+'/opts.py',log_path+'/opts.py')

n_rows = args.x_end - args.x_start
n_cols = args.y_end - args.y_start
n_slices = args.z_end - args.z_start

############################################################################
# Datasets
# the files in the imagePathFolder and labelPathFolder Folder must be saved in the form 00100001_001.dcm
# with 00100001 the caseNumber and 001 the sliceNumber
# the function generatePartitionTrainAndValidFromFolder generate List of train and valid IDs from the DicomFolderPaths randomly
############################################################################


# path to the data on elonMusk OR on my mac
#if current_dir[1:5]=='home':
args.imagePathFolder = args.imagePathFolder_server
args.labelPathFolder = args.labelPathFolder_server
#elif current_dir[1:6]=='Users':
#    args.imagePathFolder = args.imagePathFolder_local
#    args.labelPathFolder = args.labelPathFolder_local

#     # args.imagePathFolder = '/Users/christian/Recherche/rechercheProjects/17-22_StrokeAndAI/data/dataToTrainReduced/ims/0'
#     # args.labelPathFolder = '/Users/christian/Recherche/rechercheProjects/17-22_StrokeAndAI/data/dataToTrainReduced/gts/0'
#
# else:
#     raise Exception("check your path to the images and labels")

# partition the data between train and validation

partition = generatePartitionTrainAndValidFromFolderRandomly(args.imagePathFolder,args.validProportion,args.shuffle_train_val, args.imageType)


############################################################################
# callbacks
############################################################################
callbacks = []

if K.backend() == 'tensorflow':
    tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True)
    callbacks.append(tensorboard)

checkpoint = ModelCheckpoint(filepath=os.path.join(os.path.dirname(save_path), 'weights.{epoch:03d}-{val_loss:.2f}.hdf5'),
                             monitor='val_loss', verbose=0,
                             save_best_only=False, mode='auto')

checkpoint = ModelCheckpoint(filepath=os.path.join(os.path.dirname(save_path), 'weights.{epoch:04d}.hdf5'),
                             verbose=0,
                             save_best_only=False, mode='auto')

callbacks.append(checkpoint)

csv_log_cback = CSVLogger(
    '{}/historyLog.csv'.format(log_path))

callbacks.append(csv_log_cback)

#
earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
callbacks.append(earlystopper)



############################################################################
# input Data Normalization
############################################################################

normalization_args = { 'normalize': args.normalize,
                       'normalization_threshold': args.normalization_threshold,
                       'histogramNormalize': args.histogramNormalize,
                       'histogramNormalize_underThreshold': args.histogramNormalize_underThreshold,
                       'histogramNormalize_strokeThreshold': args.histogramNormalize_strokeThreshold,
                       'histogramNormalize_upperThreshold': args.histogramNormalize_upperThreshold,
                       'simpleNormalize': args.simpleNormalize,
                       'addNoise': args.addNoise,
                       'meanNoiseDistribution': args.meanNoiseDistribution,
                       'noiseMultiplicationFactor': args.noiseMultiplicationFactor,
                       'gaussian_filter': args.gaussian_filter,
                       'gaussian_filter_hsize': args.gaussian_filter_hsize,
                       'gaussian_filter_sigma': args.gaussian_filter_sigma,
                       'removeSignalunderRelativeThreshold': args.removeSignalunderRelativeThreshold,
                       'removeSignalunderRelativeThreshold_cutOff': args.removeSignalunderRelativeThreshold_cutOff}



############################################################################
# Generators
############################################################################

params = {'dim': (n_rows,n_cols,n_slices),
          'crop_parameters': [args.x_start, args.x_end, args.y_start, args.y_end, args.z_start, args.z_end],
          'batch_size': args.batch_size,
          'n_classes': args.n_classes,
          'n_channels': args.n_channels,
          'imageType': args.imageType,
          'labelType': args.labelType,
          'variableType': args.variableType,
          'loss_weights': args.loss_weights}


augmentation_args = {   'flip': args.flip,
                        'rotationRangeXAxis': args.rotationRangeXAxis,
                        'rotationRangeYAxis': args.rotationRangeYAxis,
                        'rotationRangeZAxis': args.rotationRangeZAxis,
                        'zoomRange': args.zoomRange,
                        'shiftXAxisRange': args.shiftXAxisRange,
                        'shiftYAxisRange': args.shiftYAxisRange,
                        'shiftZAxisRange': args.shiftZAxisRange,
                        'stretchFactorXAxisRange': args.stretchFactorXAxisRange,
                        'stretchFactorYAxisRange': args.stretchFactorYAxisRange,
                        'stretchFactorZAxisRange': args.stretchFactorZAxisRange,
                        'shear_NormalXAxisRange': args.shear_NormalXAxisRange,
                        'shear_NormalYAxisRange': args.shear_NormalYAxisRange,
                        'shear_NormalZAxisRange': args.shear_NormalZAxisRange,
                        'maxNumberOfTransformation': args.maxNumberOfTransformation}



training_generator = DataGenerator(partition['train'], args.imagePathFolder, args.labelPathFolder, normalization_args, args.augment, augmentation_args, **params)

# augmentation_args['augment'] = args.augment_validation

validation_generator = DataGenerator(partition['validation'], args.imagePathFolder, args.labelPathFolder, normalization_args, args.augment_validation, augmentation_args, **params)

# save the validation partition cases
f = open(os.path.dirname(save_path)+'/validation_partition.txt',"w")
for p in partition['validation']:
    f.write(p + '\n')
f.close()

############################################################################
# Design model
############################################################################

model = UNet_3D(input_shape=(n_rows,n_cols,n_slices,args.n_channels), nb_labels=args.n_classes, filters=args.features, depth=args.depth, nb_bottleneck=args.nb_bottleneck,
                padding=args.padding, activation=args.activation, activation_network=args.activation_network, batch_norm=args.batch_norm, dropout_encoder=0.06, dropout_decoder=0.06, use_preact=False, use_mvn=False).create_model()


if nGPU > 1:
    parallel_model = multi_gpu_model(model, gpus=nGPU)
else:
    parallel_model = model

plot_model(model, to_file=os.path.dirname(save_path)+'/model_plot.png', show_shapes=True, show_layer_names=True)


############################################################################
# loss function
############################################################################

loss_weights = args.loss_weights
loss_name = args.loss_name

if loss_name == 'pixel':
    def lossfunc(y_true, y_pred):
        return categorical_xentropy_loss(
            y_true, y_pred, loss_weights)
elif loss_name == 'dice':
      def lossfunc(y_true, y_pred):
           return dice_loss(y_true, y_pred, loss_weights)
elif loss_name == 'dice_multi_class':
    def lossfunc(y_true, y_pred):
        return dice_loss_multi_class(y_true, y_pred, args.n_classes)
elif loss_name == 'dice_multi_class_nb':
    def lossfunc(y_true, y_pred):
        return dice_loss_multi_class_no_background(y_true, y_pred, args.n_classes)
elif loss_name == 'dice_multi_class_nb_plus_xentropy':
    def lossfunc(y_true, y_pred):
        return dice_loss_multi_class_no_background(y_true, y_pred, args.n_classes)+categorical_crossentropy(y_true, y_pred) 
elif loss_name == 'dicePlusXentropy':
    def lossfunc(y_true, y_pred):
         return dice_loss(y_true, y_pred, loss_weights)+categorical_xentropy_loss(
            y_true, y_pred, loss_weights)
elif loss_name == 'dicePlusXentropyWeighted':
    def lossfunc(y_true, y_pred):
        return 0.00008*dice_loss(y_true, y_pred, loss_weights)+0.99992*categorical_xentropy_loss(
            y_true, y_pred, loss_weights)
elif loss_name == 'jaccard':
    def lossfunc(y_true, y_pred):
        return jaccard_loss(y_true, y_pred, loss_weights)
elif loss_name == 'euclideanDistance':
    def lossfunc(y_true, y_pred):
        return euclidean_distance_loss(y_true, y_pred)
elif loss_name == 'euclideanDistancePlusXentropyWeighted':
    def lossfunc(y_true, y_pred):
        return 0.0001 * euclidean_distance_loss(y_true, y_pred) + 0.9999 * categorical_xentropy_loss(y_true, y_pred, loss_weights)
else:
    raise Exception("Unknown loss ({})".format(loss_name))



############################################################################
# compile and fit the model
############################################################################

metrics = [dice_coef_background, dice_coef_first_label, dice_coef_second_label, dice_coef_third_label, dice_coef_fourth_label]

#metrics = [dice_coef_background, dice_coef_first_label, dice_coef_second_label]

if nGPU > 1:

    parallel_model.compile(optimizer=args.optimizer_type,
                           loss=lossfunc,
                           metrics=metrics)

    results = parallel_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=args.use_multiprocessing,
                    # use_multiprocessing=True,
                    epochs=args.epochs,
                    callbacks = callbacks)

else:

    model.compile(optimizer=args.optimizer_type,
                           loss=lossfunc,
                           metrics=metrics)

    results = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  use_multiprocessing=args.use_multiprocessing,
                                  # use_multiprocessing=True,
                                  epochs=args.epochs,
                                  callbacks=callbacks)



############################################################################
# save
############################################################################

# parallel model cannot save!!!
# model.compile(optimizer=args.optimizer_type,
#               loss=lossfunc,
#               metrics=metrics)

# model.set_weights(parallel_model.get_weights())


# save arguments
f = open(os.path.dirname(save_path)+'/args.txt',"w")
for k, v in vars(args).items():
    f.write("{} = {}\n".format(k, v))
f.close()

# save model
model.model.save(save_path)

# this is not working in Pycharm on ElonMusk: `pydot` failed to call GraphViz.
#plot_learning_curve(results,os.path.dirname(save_path))




############################################################################
# predict
############################################################################

#
# model = UNet_3D(input_shape=(args.n_rows,args.n_cols,args.n_slices,args.n_channels), nb_labels=args.n_classes, filters=args.features, depth=args.depth, nb_bottleneck=args.nb_bottleneck,
#                 padding=args.padding, activation=args.activation, activation_network=args.activation_network, batch_norm=args.batch_norm, dropout_encoder=0.0, dropout_decoder=0.0, use_preact=False, use_mvn=False).create_model()
#
#
# model.compile(optimizer=args.optimizer_type,
#               loss=lossfunc,
#               metrics=metrics)
#               # metrics = ['accuracy'])
#
# # model = multi_gpu_model(model, gpus=1)
#
#
#
#
# if current_dir[1:5]=='home':
#     datatest_ims_path = args.imageTestPathFolder_server
#     datatest_gts_path = args.labelTestPathFolder_server
# elif current_dir[1:6]=='Users':
#     datatest_ims_path = args.imageTestPathFolder_local
#     datatest_gts_path = args.labelTestPathFolder_local
#
#
# PredictionFolder_path = os.path.dirname(save_path) + '/predictions'
# # predict_3D(model, datatest_ims_path, datatest_gts_path, PredictionFolder_path)
# predict_3D(model, datatest_ims_path, datatest_gts_path, PredictionFolder_path, args)
