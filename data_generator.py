from tensorflow.keras import backend as K

import numpy as np
import tensorflow.keras as keras
import pydicom
import glob
from PIL import Image
from augmentation import augmentor
import random
import datetime
import os
import matplotlib.pyplot as plt
import nibabel as nib
from tensorflow.keras.utils import to_categorical, Sequence
from skimage.transform import resize
from util import simpleNormalization, addNoise, intensityNormalization, CTNormalization

class DataGenerator(Sequence):

    def __init__(self, 
                list_IDs, 
                imagePath, 
                labelPath, 
                normalization_args, 
                augment = 0, 
                augmentation_args = {}, 
                preload_data = False, 
                imageType = '.dcm', 
                labelType = '.dcm', 
                batch_size=32, 
                dim=(32, 32, 32), 
                crop_parameters=[0,10,0,10,0,10], 
                resize_parameters=None,
                pad=False,
                n_channels=1, 
                n_classes=10, 
                shuffle=True, 
                variableTypeX = 'float32', 
                variableTypeY = 'float32', 
                savePath = None, 
                preload=False,
                **kwargs):

        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.imagePath = imagePath
        self.labelPath = labelPath
        self.imageType = imageType
        self.labelType = labelType
        self.variableTypeX = variableTypeX
        self.variableTypeY = variableTypeY
        self.crop_parameters = crop_parameters
        self.resize_parameters = resize_parameters
        self.pad = pad

        self.augment = augment
        self.augmentation_args = augmentation_args
        self.normalization_args = normalization_args
        self.save_path = savePath
        self.numberOfAufmentationImageSaved = 0

        #preload data into RAM if stated
        self.preload = preload_data
        if self.preload:
            self.dataX = self.loadData(list_IDs, self.imagePath, order=1)
            self.dataY = self.loadData(list_IDs, self.labelPath, order=0)

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


    def loadData(self, IDs, folderPath, imageType, order):
        # X: we need to add a new axis for the channel dimension
        # Y: we need to transform to categorical
        #generate dict with sample-ID as key and dataobj as value
        dataDict = {}
        for i, ID in enumerate(IDs):
            if '.nii.gz' in self.imageType or '.nii' in self.imageType:
                dataObj = self.load3DImagesNii(folderPath, ID, imageType, self.crop_parameters, self.resize_parameters,order=order)[..., np.newaxis]
            elif self.imageType == '.dcm':
                dataObj = self.load3DImagesDcm(folderPath, ID, self.crop_parameters, self.resize_parameters, order=order)[..., np.newaxis]
            elif self.imageType == '.npy':
                pathToNpy = folderPath + '/' + ID + '.npy'
                dataObj = np.load(pathToNpy)[..., np.newaxis]
            else:
                raise Exception("Unknown image type")
        
            #add new key-value pair
            dataDict[ID] = dataObj

        return dataDict

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.list_IDs))

        if self.shuffle == True:
            np.random.shuffle(self.indices)
        
        self.n = 0

        #Print mean number of augmentations per input sample
        print('; {:.1f} augmentations performed on average'.format(self.brainAugmentor.meanNumberTransforms()))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        #Load data at this point if not preloaded already
        if not self.preload:
            self.dataX = self.loadData(list_IDs_temp, self.imagePath, self.imageType, order=1)
            self.dataY = self.loadData(list_IDs_temp, self.labelPath, self.labelType, order=0)

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

            #preprocessing of Y
            #-------------------
            # add noise
            if (self.normalization_args['addNoise'] == 1):
                X_temp = addNoise(X_temp, self.normalization_args['normalization_threshold'], self.normalization_args['meanNoiseDistribution'], self.normalization_args['noiseMultiplicationFactor'])

            # The intensity augmentation can only be used WITHOUT prior rescaling to [0,1] of [-1,1]!
            if (self.normalization_args['intensityNormalize'] == 1):
                X_temp = intensityNormalization(X_temp, augment=self.augment)

            # simpleNormalization
            if (self.normalization_args['simpleNormalize'] == 1):
                X_temp = simpleNormalization(X_temp)

            # CTNormalization
            if (self.normalization_args['ctNormalize'] == 1):
                X_temp = CTNormalization(X_temp)

            #2D data: slice selection
            if len(self.dim) < 3:
                randint = np.random.randint(0, X_temp.shape[-2], self.batch_size)
                X_temp = X_temp[...,randint[i],:]
                Y_temp = Y_temp[...,randint[i],:]

           #assign final data
            X[i,] = X_temp
            Y[i,] = to_categorical(Y_temp[...,0], num_classes=self.n_classes, dtype=self.variableTypeY)

        return X.astype(self.variableTypeX), Y.astype(self.variableTypeY)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        # Find list of IDs
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end()
            self.n = 0

        return data
        
    def load3DImagesNii(self, pathToNiiFolder, caseNumber, imageType, crop_params, resize_parameters=None, order=None):
        pathToNiiGz = pathToNiiFolder + '/' + caseNumber + imageType
        image_array = np.asanyarray(nib.load(pathToNiiGz).dataobj).squeeze()

        if resize_parameters is not None:
            if order == 0:
                image_array = resize(image_array, resize_parameters, order=order, anti_aliasing=False, preserve_range=True)
            else:
                image_array = resize(image_array, resize_parameters, order=order, anti_aliasing=False, preserve_range=True)

        if self.pad:
            padding = np.asarray(self.dim) // 2
            padding -= [(crop_params[1] - crop_params[0]) // 2, (crop_params[3] - crop_params[2]) // 2, (crop_params[5] - crop_params[4]) // 2]
            image_array = np.pad(image_array, np.transpose([padding, padding]), mode='constant')[crop_params[0]: crop_params[1], crop_params[2]: crop_params[3], crop_params[4]: crop_params[5]]
        else:
            image_array = image_array[crop_params[0]: crop_params[1], crop_params[2]: crop_params[3], crop_params[4]: crop_params[5]]
            
        return image_array


    # load the DICOM files from a Folder, and return a 3 dim numpy array in LPS orientation
    def load3DImageDCM_LPS(self, pathToLoadDicomFolder, orientation):
    
        fileTuples = []

        for fname in glob.glob(pathToLoadDicomFolder, recursive=False):
            fileTuples.append((fname,pydicom.dcmread(fname)))

        # sort the tuple (filename, dicom image) by image position
            fileTuples = sorted(fileTuples, key=lambda s: s[1].ImagePositionPatient[orientation])

        files = [x[1] for x in fileTuples]
        fileNames = [x[0] for x in fileTuples]

        # print("PATH TO LOAD DICOM FOLDER: {}".format(pathToLoadDicomFolder))
        # print(fileNames)

        img2d_shape = list(files[0].pixel_array.shape)

        if orientation == 0:
            L = len(files)
            P = img2d_shape[1] 
            S = img2d_shape[0] #in fact -
        if orientation == 1:
            L = img2d_shape[1]
            P = len(files)
            S = img2d_shape[0] #in fact -
            
        if orientation == 2:
            L = img2d_shape[1]
            P = img2d_shape[0]
            S = len(files)
        

        img_shape = [L,P,S]
        # print("image shape in LPS: {}".format(img_shape))

        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the files
        for i, s in enumerate(files):
            img2d = s.pixel_array

            img_shape = [L,P,S]
            
            if orientation == 0:
                img3d[i, :, :] = img2d.T
            
            if orientation == 1:
                img2d = np.flip(img2d,axis=0)
                img3d[:, i, :] = img2d.T
        
            if orientation == 2:
                img2d = np.flip(img2d,axis=0)
                img3d[:, :, i] = img2d.T
                            
        return img3d



    def load3DImagesDcm(self, pathToDcmFolder, ID, crop_params, resize_parameters=None, order=None):
        pathToDicomFolder = pathToDcmFolder + '/' + ID
        orientation = get_orientation(pathToDicomFolder)
        image_array = self.load3DImageDCM_LPS(pathToDicomFolder+'/*', orientation)


        if resize_parameters is not None:
            if order == 0:
                image_array = resize(image_array, resize_parameters, order=order, anti_aliasing=False, preserve_range=True)
            else:
                image_array = resize(image_array, resize_parameters, order=order, anti_aliasing=True, preserve_range=True)

        if self.pad:
            padding = np.asarray(self.dim) // 2
            padding -= [(crop_params[1] - crop_params[0]) // 2, (crop_params[3] - crop_params[2]) // 2, (crop_params[5] - crop_params[4]) // 2]

            print(self.dim, padding)
            image_array = np.pad(image_array, np.transpose([padding, padding]), mode='constant')[crop_params[0]: crop_params[1], crop_params[2]: crop_params[3], crop_params[4]: crop_params[5]]
        else:
            image_array = image_array[crop_params[0]: crop_params[1], crop_params[2]: crop_params[3], crop_params[4]: crop_params[5]]
            

        return image_array
    


#####################################
# Helper functions
#####################################

def listOfCasesInFolder(pathToFolder, image_type='.dcm'):

    listOfCases = []
    listOfFiles = [f for f in os.listdir(pathToFolder) if not f.startswith('.')]#os.listdir(pathToFolder)
    
    if('.dcm' in image_type):
        listOfCases = listOfFiles

    if('.nii.gz' in image_type or '.npy' in image_type):
        for f in listOfFiles:
            if f.endswith(image_type) and f[0] != '.':
                listOfCases.append(f.split('.')[0])

    return list(listOfCases)


def removeStrokeBelowThreshold(listOfCases, labelsPathFolder, image_type='.dcm', threshold=0, n_classes=1):
    
    valid = []
    for case in listOfCases:
        try:
            gts = nib.load(labelsPathFolder + '/' + case + image_type).get_fdata()
            if threshold >= 0 and ( np.sum(gts==n_classes-1) >= threshold or np.sum(gts==255) >= threshold):
                valid.append(case)
            elif threshold < 0 and np.sum(gts == n_classes-1) < -threshold:
                valid.append(case)
        #If not gts file available -> normal database
        except FileNotFoundError:
            valid.append(case)

    return valid

def get_id_lists(imagePathFolder, _validProportion, shuffle_train_val, image_type='.dcm', labelPathFolder=None, label_type='.nii.gz', threshold=0, n_classes=1):
    # generate List of train and valid IDs from the DicomFolderPaths randomly

    _listOfCases = listOfCasesInFolder(imagePathFolder, image_type)

    if labelPathFolder is not None:
        _listOfCases = removeStrokeBelowThreshold(_listOfCases, labelPathFolder, image_type=label_type, threshold=threshold, n_classes=n_classes)

    index = np.arange(len(_listOfCases))

    if shuffle_train_val:
        np.random.shuffle(index)

    validNumbers = int(np.floor(len(_listOfCases) * _validProportion))

    indexValid = random.sample(range(0, len(_listOfCases)), validNumbers)
    indexTrain = []
    for k in index:
        if k not in indexValid:
            indexTrain.append(k)

    _listOfTrainCasesID = [_listOfCases[k] for k in indexTrain]
    _listOfValidCasesID = [_listOfCases[k] for k in indexValid]

    partition = {'train': _listOfTrainCasesID, 'validation': _listOfValidCasesID}

    return partition



def get_orientation(dcm_path):
    """
    Function which returns the (encoded) orientation of a dicom volume.

    Input: Path to dicom volume

    Returns:  0 (if the images are sagittal slices), 1 (if the images are coronal slices), 2 (if the images are axial slices).
    """

    orientations = []

    for file in os.listdir(dcm_path):
        
        slc = dcm_path + '/' + file
        slc = pydicom.dcmread(slc)

        dicom_orientation = np.asarray(slc.ImageOrientationPatient).reshape((2,3))
        dicom_orientation_argmax = sorted(np.argmax(np.abs(dicom_orientation), axis=1))

        orientations.append(np.delete([0,1,2], sorted(dicom_orientation_argmax)))

    assert len(np.unique(orientations)) == 1, "ERROR: Not all slices have same orientation!"

    return np.unique(orientations)[0]#, np.asarray([np.sign(dicom_orientation[i][dicom_orientation_argmax[i]]) for i in range(2)], dtype=int)

