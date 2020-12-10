from tensorflow.keras import backend as K

import numpy as np
import tensorflow.keras as keras
import pydicom
import glob
from PIL import Image
from augmentation import augmentor
from util import simpleNormalization, addNoise, intensityNormalization, CTNormalization
import datetime
import time
import os
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize

import threading
import queue

#--------------------
#SETTINGS
#--------------------
NUM_THREADS = 3
MAX_LENGTH_QUEUE = 3 # +NUM_THREADS samples will be loaded and wait to put if sampleQueue is full
#--------------------

class DataGenerator_MT(keras.utils.Sequence):
    def __init__(self, 
                 epochs, 
                 list_IDs, 
                 imagePathFolder, 
                 labelPathFolder, 
                 normalization_args, 
                 augment, 
                 augmentation_args, 
                 preload_data = False, 
                 imageType = '.dcm', 
                 labelType = '.dcm', 
                 batch_size=32, dim=(32, 32, 32), 
                 crop_parameters=[0,10,0,10,0,10], 
                 resize_parameters=None,
                 n_channels=1, 
                 n_classes=10, 
                 shuffle=True, 
                 variableTypeX = 'float32', 
                 variableTypeY = 'float32', 
                 **kwargs):

        #params for sample identification & ordering
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.batch_size = batch_size

        #create augmentor param set
        augmentation_params = {
                    'rotationAngle_Xaxis': augmentation_args['rotationRangeXAxis'],
                    'rotationAngle_Yaxis': augmentation_args['rotationRangeYAxis'],
                    'rotationAngle_Zaxis': augmentation_args['rotationRangeZAxis'],
                    'shift_Xaxis': augmentation_args['shiftXAxisRange'],
                    'shift_Yaxis': augmentation_args['shiftYAxisRange'],
                    'shift_Zaxis': augmentation_args['shiftZAxisRange'],
                    'stretch_Xaxis': augmentation_args['stretchFactorXAxisRange'],
                    'stretch_Yaxis': augmentation_args['stretchFactorYAxisRange'],
                    'stretch_Zaxis': augmentation_args['stretchFactorZAxisRange'],
                    'shear_NormalXAxis': augmentation_args['shear_NormalXAxisRange'],
                    'shear_NormalYAxis': augmentation_args['shear_NormalYAxisRange'],
                    'shear_NormalZAxis': augmentation_args['shear_NormalZAxisRange'],
                    'zoom': augmentation_args['zoomRange'],
                    'flip': augmentation_args['flip'],
                    }

        #create data loader instance (where samples will be prepared for training)
        self.dataLoader = DataLoader(epochs, list_IDs, imagePathFolder, labelPathFolder, normalization_args, augment, augmentation_params, augmentation_args['maxNumberOfTransformation'], imageType, labelType, dim, crop_parameters, resize_parameters, n_channels, n_classes, variableTypeX, variableTypeY)

        #set index order for first epoch
        self.epoch_count = 0
        self.on_epoch_end()


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        self.epoch_count += 1

        #transmit new indexes to DataLoader and start loading
        self.dataLoader.startLoading(self.indexes, self.batch_size, self.epoch_count)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'

        #get 'batch_size' samples from DataLoader
        X, Y = [], []
        for i in range(self.batch_size):
            X_temp, Y_temp = self.dataLoader.getLoadedSample()
            X.append(X_temp)
            Y.append(Y_temp)

        #concatenate to batch
        X = np.concatenate(X, axis = 0)
        Y = np.concatenate(Y, axis = 0)

        return X, Y


#--------------------------------------------------------------------
# DataLoader class handles loaded and preprocessed data for training
#--------------------------------------------------------------------
class DataLoader():
    def __init__(self, epochs, list_IDs, imagePathFolder, labelPathFolder, normalization_args, augment, augmentation_params, numMaxTransforms, imageType, labelType, dim, crop_parameters, resize_parameters, n_channels, n_classes, variableTypeX, variableTypeY):

        #general settings
        self.maxEpochs = epochs
        self.list_IDs = list_IDs
        self.imagePathFolder = imagePathFolder
        self.labelPathFolder = labelPathFolder
        self.normalization_args = normalization_args
        self.augment = augment
        self.augmentation_params = augmentation_params
        self.numMaxTransforms = numMaxTransforms
        self.imageType = imageType
        self.labelType = labelType
        self.dim = dim
        self.crop_parameters = crop_parameters
        self.resize_parameters = resize_parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.variableTypeX = variableTypeX
        self.variableTypeY = variableTypeY

        #define queue for indexes & loaded and preprocessed samples
        self.indexQueue = queue.Queue()
        self.sampleQueue = queue.Queue()
        self.indexQLock = threading.Lock()
        self.sampleQLock = threading.Lock()

        #define event when queue is full
        self.queueSemaphore = threading.Semaphore(value=MAX_LENGTH_QUEUE)

        #threads
        self.threads = []


    def startLoading(self, indexes, batch_size, epoch):
        #wait for all threads to finish and clear list
        for thd in self.threads:
            thd.join()
        self.threads.clear()

        #remove remaining samples (if any)
        self.sampleQLock.acquire()
        while not self.sampleQueue.empty():
            self.sampleQueue.get()
            self.queueSemaphore.release()
        self.sampleQLock.release()

        #terminate if last epoch
        if epoch > self.maxEpochs:
            return

        #fill index queue with new index ordering
        #------------------------------------------
        self.indexQLock.acquire()
        for idx in indexes:
            self.indexQueue.put(idx)

        #add indexes of first batch 2 times (because __getitem__ is (always?) accessed 2 times with index=0)
        for i in range(batch_size):
            self.indexQueue.put(indexes[i])

        self.indexQLock.release()
        #------------------------------------------

        #create new threads for image loading and preprocessing
        for thID in range(NUM_THREADS):
            self.threads.append(LoadingThread(thID, self.list_IDs, self.imagePathFolder, self.labelPathFolder, self.normalization_args, self.augment, self.augmentation_params, self.numMaxTransforms,
                                             self.imageType, self.labelType, self.dim, self.crop_parameters, self.resize_parameters, self.n_channels, self.n_classes, self.variableTypeX, self.variableTypeY,
                                             self.indexQueue, self.sampleQueue, self.indexQLock, self.sampleQLock, self.queueSemaphore))
        #start threads
        for thread in self.threads:
            thread.start()


    def getLoadedSample(self):
        #wait for sample
        while self.sampleQueue.empty():
            pass

        #get sample
        self.sampleQLock.acquire()
        X, Y = self.sampleQueue.get()
        self.queueSemaphore.release()
        self.sampleQLock.release()

        return X, Y


#-------------------------------------------------------------------
# WorkerThread class fills queue with preprocessed training samples
#-------------------------------------------------------------------
class LoadingThread(threading.Thread):
    def __init__(self, threadID, list_IDs, imagePathFolder, labelPathFolder, normalization_args, augment, augmentation_params, numMaxTransforms, imageType, labelType, dim, crop_parameters, resize_parameters, n_channels, n_classes, variableTypeX, variableTypeY, iQueue, sQueue, iQLock, sQLock, qSemaphore):
        threading.Thread.__init__(self)

        #ID
        self.threadID = str(threadID)

        #general settings
        self.list_IDs = list_IDs
        self.imgPath = imagePathFolder
        self.labelPath = labelPathFolder
        self.normalization_args = normalization_args
        self.augment = augment
        self.numMaxTransforms = numMaxTransforms
        self.imgType = imageType
        self.labelType = labelType
        self.dim = dim
        self.crop_params = crop_parameters
        self.resize_parameters = resize_parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.varTypeX = variableTypeX
        self.varTypeY = variableTypeY

        #queue and lock instances
        self.indexQueue = iQueue
        self.sampleQueue = sQueue
        self.indexQLock = iQLock
        self.sampleQLock = sQLock
        self.queueSemaphore = qSemaphore

        #augmentor
        self.brainAugmentor = augmentor(**augmentation_params)


    def run(self):
        print("Start Generator Thread " + self.threadID)

        #loop until no more samples to process
        while not self.indexQueue.empty():

            #get index to process (if empty in the meantime, release lock and return)
            try:
                self.indexQLock.acquire()
                ID = self.list_IDs[self.indexQueue.get()]
                self.indexQLock.release()
            except queue.Empty:
                self.indexQLock.release()
                return

            #load and preprocess
            dataTuple = self.dataGeneration(ID)

            #wait for semaphore and put to queue
            self.queueSemaphore.acquire()
            self.sampleQLock.acquire()
            self.sampleQueue.put(dataTuple)
            self.sampleQLock.release()

        print ("Exit Generator Thread " + self.threadID)


    def loadData(self, ID, folderPath, imgType):
        if '.nii' in imgType:
            dataObj = self.load3DImagesNii(folderPath, ID, imgType, self.crop_params, self.resize_parameters)[..., np.newaxis]
        elif '.dcm' in imgType:
            dataObj = self.load3DImagesDcm(folderPath, ID, imgType, self.crop_params, self.resize_parameters)[..., np.newaxis]
        else:
            raise Exception("Unknown image type")

        return dataObj


    def dataGeneration(self, ID):
        'Generates data containing one sample of batch'
        #Load data
        X_temp = self.loadData(ID, self.imgPath, self.imgType)
        Y_temp = self.loadData(ID, self.labelPath, self.labelType)

        #augmentation
        #----------------------------------------------------------------------
        if self.augment:
            X_temp, Y_temp = self.brainAugmentor.augmentXY(X_temp, Y_temp, self.numMaxTransforms)


        #preprocessing of X
        #-------------------
        # add noise
        if (self.normalization_args['addNoise'] == 1):
            X_temp = addNoise(X_temp, self.normalization_args['normalization_threshold'], self.normalization_args['meanNoiseDistribution'], self.normalization_args['noiseMultiplicationFactor'])

        # normalization
        if (self.normalization_args['normalize'] == 1):
            X_temp = normalize(X_temp, self.normalization_args['normalization_threshold'])

        # simpleNormalization
        if (self.normalization_args['simpleNormalize'] == 1):
            X_temp = simpleNormalization(X_temp)

        # The intensity augmentation can only be used WITHOUt prior rescaling to [0,1] of [-1,1]!
        elif (self.normalization_args['intensityNormalize'] == 1):
            X_temp = intensityNormalization(X_temp, augment=self.augment)

        # CTNormalization
        if (self.normalization_args['ctNormalize'] == 1):

            X_temp = CTNormalization(X_temp)

        # gaussian filter
        if (self.normalization_args['gaussian_filter'] == 1):
            X_temp = gaussian_smoothing(X_temp, self.normalization_args['gaussian_filter_hsize'], self.normalization_args['gaussian_filter_sigma'])

        #assign final data (add additional axis at index 0 for batch_dimension)
        #----------------------------------------------------------------------
        X = X_temp[np.newaxis, ...]
        Y = keras.utils.to_categorical(Y_temp[..., 0], num_classes=self.n_classes, dtype=self.varTypeY)[np.newaxis, ...]

        return X.astype(self.varTypeX), Y.astype(self.varTypeY)


    def load3DImagesNii(self, pathToNiiFolder, caseNumber, imageType, crop_params, resize_parameters):
        pathToNiiGz = pathToNiiFolder + '/' + caseNumber + imageType
        image_array = np.asanyarray(nib.load(pathToNiiGz).dataobj)
        
        if resize_parameters is not None:
            image_array = resize(image_array, resize_parameters)

        return image_array[crop_params[0]: crop_params[1], crop_params[2]: crop_params[3], crop_params[4]: crop_params[5]]


    # load the DICOM files from a Folder, and return a 3 dim numpy array in LPS orientation
    def load3DImageDCM_LPS(self, pathToLoadDicomFolder, orientation):

        # print("PATH TO LOAD DICOM FOLDER: {}".format(pathToLoadDicomFolder))
        fileTuples = []

        # print('glob: {}'.format(pathToLoadDicomFolder))
        for fname in glob.glob(pathToLoadDicomFolder, recursive=False):
            fileTuples.append((fname,pydicom.dcmread(fname)))

        # sort the tuple (filename, dicom image) by image position
        if orientation == 1:
            fileTuples = sorted(fileTuples, key=lambda s: s[1].ImagePositionPatient[2])

        if orientation == 2:
            fileTuples = sorted(fileTuples, key=lambda s: s[1].ImagePositionPatient[0])

        if orientation == 3:
            fileTuples = sorted(fileTuples, key=lambda s: fileTuples[1].ImagePositionPatient[1])

        files = [x[1] for x in fileTuples]
        fileNames = [x[0] for x in fileTuples]

        # print("PATH TO LOAD DICOM FOLDER: {}".format(pathToLoadDicomFolder))
        # print(fileNames)

        img2d_shape = list(files[0].pixel_array.shape)

        if orientation == 1:
            L = img2d_shape[1]
            P = img2d_shape[0]
            S = len(files)

        if orientation == 2:
            L = len(files)
            P = img2d_shape[1]
            S = img2d_shape[0] #in fact -

        if orientation == 3:
            L = img2d_shape[1]
            P = len(files)
            S = img2d_shape[0] #in fact -

        img_shape = [L,P,S]
        # print("image shape in LPS: {}".format(img_shape))

        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the files
        for i, s in enumerate(files):
            img2d = s.pixel_array

            img_shape = [L,P,S]

            if orientation == 1:
                img3d[:, :, i] = img2d.T

            if orientation == 2:
                img2d = np.flip(img2d,axis=0)
                img3d[i, :, :] = img2d.T

            if orientation == 3:
                img2d = np.flip(img2d,axis=0)
                img3d[:, i, :] = img2d.T

        return img3d


    def getOrientation(self, pathToLoadDicomFolder):
        # return 1 if  axial, 2 if sagittal, and 3 if coronal
        orientation = os.popen("orientation -nonVerbose " + pathToLoadDicomFolder).read()
        orientation = int(orientation[0])
        return orientation


    def load3DImagesDcm(self, pathToDcmFolder, ID, crop_params, resize_parameters):
        pathToDicomFolder = pathToDcmFolder + '/' + ID
        orientation = self.getOrientation(pathToDicomFolder)
        image_array = self.load3DImageDCM_LPS(pathToDicomFolder+'/*', orientation)

        if resize_parameters is not None:
            image_array = resize(image_array, resize_parameters)

        return image_array[crop_params[0]: crop_params[1], crop_params[2]: crop_params[3], crop_params[4]: crop_params[5]]
