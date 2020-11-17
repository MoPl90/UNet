from keras import backend as K

import numpy as np
import keras as keras
import pydicom
import glob
from PIL import Image
from augmentation import augmentor
import random
from util import simpleNormalization, addNoise
import datetime
import os
import matplotlib.pyplot as plt
import nibabel as nib

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, imagePathFolder, labelPathFolder, normalization_args, augment, augmentation_args, preload_data = False, imageType = '.dcm', labelType = '.dcm', batch_size=32, dim=(32, 32, 32), crop_parameters=[0,10,0,10,0,10], n_channels=1, n_classes=10, shuffle=True, variableTypeX = 'float32', variableTypeY = 'float32', savePath = None, preload=False, spade_norm=False, lr_seg=False, **kwargs):

        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.imagePathFolder = imagePathFolder
        self.labelPathFolder = labelPathFolder
        self.imageType = imageType
        self.labelType = labelType
        self.variableTypeX = variableTypeX
        self.variableTypeY = variableTypeY
        self.crop_parameters = crop_parameters

        self.augment = augment
        self.augmentation_args = augmentation_args
        self.normalization_args = normalization_args
        self.save_path = savePath
        self.numberOfAufmentationImageSaved = 0

        #preload data into RAM if stated
        self.preload = preload_data
        if self.preload:
            self.dataX = self._preloadData(list_IDs, self.imagePathFolder, self.crop_parameters, self.imageType)
            self.dataY = self._preloadData(list_IDs, self.labelPathFolder, self.crop_parameters, self.labelType, labels=True)

        #create augmentor
        augmentation_parameters = {
                    'rotationAngle_Xaxis': self.augmentation_args['rotationRangeXAxis'],
                    'rotationAngle_Yaxis': self.augmentation_args['rotationRangeYAxis'],
                    'rotationAngle_Zaxis': self.augmentation_args['rotationRangeZAxis'],
                    'shift_Xaxis': self.augmentation_args['shiftXAxisRange'],
                    'shift_Yaxis': self.augmentation_args['shiftYAxisRange'],
                    'shift_Zaxis': self.augmentation_args['shiftZAxisRange'],
                    'stretch_Xaxis': self.augmentation_args['stretchFactorXAxisRange'],
                    'stretch_Yaxis': self.augmentation_args['stretchFactorYAxisRange'],
                    'stretch_Zaxis': self.augmentation_args['stretchFactorZAxisRange'],
                    'shear_NormalXAxis': self.augmentation_args['shear_NormalXAxisRange'],
                    'shear_NormalYAxis': self.augmentation_args['shear_NormalYAxisRange'],
                    'shear_NormalZAxis': self.augmentation_args['shear_NormalZAxisRange'],
                    'zoom': self.augmentation_args['zoomRange'],
                    'flip': self.augmentation_args['flip'],
                    }

        self.brainAugmentor = augmentor(**augmentation_parameters)

        #set index order for first epoch
        self.on_epoch_end()


    def _preloadData(self, IDs, folderPath, crop_params, imgType, labels=False):
        # X: we need to add a new axis for the channel dimension
        # Y: we need to transform to categorical
        #generate dict with sample-ID as key and dataobj as value
        dataDict = {}
        for i, ID in enumerate(IDs):
            #load data
            dataObj = self.load3DImagesNii(folderPath, ID, imgType, crop_params)[..., np.newaxis]

            #add new key-value pair
            dataDict[ID] = dataObj

        return dataDict


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        #Print mean number of augmentations per input sample
        print('; {:.1f} augmentations performed on average'.format(self.brainAugmentor.meanNumberTransforms()))


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        #Load data at this point if not preloaded already
        if not self.preload:
            self.dataX = self._preloadData(list_IDs_temp, self.imagePathFolder, self.crop_parameters, self.imageType)
            self.dataY = self._preloadData(list_IDs_temp, self.labelPathFolder, self.crop_parameters, self.labelType, labels=True)

        #Generate data
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_classes))
        for i, ID in enumerate(list_IDs_temp):
            #get according data
            X_temp = self.dataX[ID]
            Y_temp = self.dataY[ID]

            #augmentation
            #-------------------
            if (self.augment == 1):
                X_temp, Y_temp = self.brainAugmentor.augmentXY(X_temp, Y_temp, self.augmentation_args['maxNumberOfTransformation'])

            #preprocessing of X
            #-------------------
            # add noise
            if (self.normalization_args['addNoise'] == 1):
                X_temp = addNoise(X_temp, self.normalization_args['normalization_threshold'], self.normalization_args['meanNoiseDistribution'], self.normalization_args['noiseMultiplicationFactor'])

            # normalization
            if (self.normalization_args['normalize'] == 1):
                X_temp = normalize(X_temp, self.normalization_args['normalization_threshold'])
            """
            if (self.normalization_args['histogramNormalize'] == 1):
                X_temp = histogramNormalize(X_temp, self.normalization_args['histogramNormalize_underThreshold'], self.normalization_args['histogramNormalize_strokeThreshold'], self.normalization_args['histogramNormalize_upperThreshold'])

                if (self.normalization_args['removeSignalunderRelativeThreshold']):
                    X_temp = removeSignalunderRelativeThreshold(X_temp, self.normalization_args['removeSignalunderRelativeThreshold_cutOff'])
            """
            # simpleNormalization
            if (self.normalization_args['simpleNormalize'] == 1):
                X_temp = simpleNormalization(X_temp)

            # # CTNormalization
            # if (self.normalization_args['ctNormalize'] == 1):
            #     X_temp = CTNormalization(X_temp)

            # gaussian filter
            if (self.normalization_args['gaussian_filter'] == 1):
                X_temp = gaussian_smoothing(X_temp, self.normalization_args['gaussian_filter_hsize'], self.normalization_args['gaussian_filter_sigma'])

            #assign final data
            #--------------------
            X[i,] = X_temp
            Y[i,] = keras.utils.to_categorical(Y_temp[...,0], num_classes=self.n_classes, dtype=self.variableTypeY)

        return X.astype(self.variableTypeX), Y.astype(self.variableTypeY)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        # Find list of IDs
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def load3DImagesNii(self, pathToNiiFolder, caseNumber, imageType, crop_parameters):
        pathToNiiGz = pathToNiiFolder + '/' + caseNumber + imageType
        image_array = np.asanyarray(nib.load(pathToNiiGz).dataobj)
        return image_array[crop_parameters[0]: crop_parameters[1], crop_parameters[2]: crop_parameters[3], crop_parameters[4]: crop_parameters[5]]
