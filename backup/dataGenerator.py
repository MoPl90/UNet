from keras import backend as K

import numpy as np
import keras
import pydicom
import glob
from PIL import Image
from augmentation import augment
import random
from util import normalize, addNoise, gaussian_smoothing, histogramNormalize, removeSignalunderRelativeThreshold,simpleNormalization
import datetime
import os
import matplotlib.pyplot as plt
import nibabel as nib


class DataGenerator(keras.utils.Sequence):
# class DataGenerator():


    def __init__(self, list_IDs, imagePathFolder, labelPathFolder, normalization_args, augment, augmentation_args, imageType = '.dcm', labelType = '.dcm', batch_size=32, dim=(32, 32, 32), crop_parameters=[0,128,0,128,4,36], n_channels=1, n_classes=10, shuffle=True, variableType = 'float32', **kwargs):
        'Initialization'
        self.dim = dim
        self.crop_parameters = crop_parameters
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
        self.variableType = variableType
        self.on_epoch_end()
        self.augment = augment
        self.augmentation_args = augmentation_args
        self.normalization_args = normalization_args
        self.numberOfAufmentationImageSaved = 0
        # self.mask

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def saveAugmentationImage(self, oldX, oldY, X, Y, imageNumber):
    # def saveAugmentationImage(self,X):

        # for j, sliceNumber in enumerate(_listOfSlicesNumbers):

        savePathDataGeneration = '/home/fjia/jiaf_4_3d_unet/4_3D_Unet/DataGenerationSample_100'
        if os.path.exists(savePathDataGeneration) is False:
            os.makedirs(savePathDataGeneration)

        name = 'raw image'
        fig = plt.figure(figsize=(14, 10))
        ax1 = fig.add_subplot(221)
        # ax1.imshow(oldX[i,:, :, 6, 0])
        ax1.imshow(oldX[0, :, :, 18, 0])
        ax1.title.set_text(name)

        removeThisValueFromHistogramToSeeBetter = 0
        _oldX = oldX[~(oldX == removeThisValueFromHistogramToSeeBetter)]
        ax2 = fig.add_subplot(222)
        ax2.hist(_oldX, 300)
        # ax2.set_title(name)
        ax2.set_title(' median= %.1f' % np.median(_oldX) + ' mean= %.1f' % np.mean(_oldX) + ' max= %.1f' % np.max(_oldX) + ' min= %.1f' % np.min(_oldX))

        # print('median ' + name + ' = %.4f' % np.median(_oldX))
        # print('mean  ' + name + ' = %.4f' % np.mean(_oldX))
        # print('max  ' + name + ' = %.4f' % np.max(_oldX))
        # print('min  ' + name + ' = %.4f' % np.min(_oldX))

        name = 'after normalization and augmentation'
        ax3 = fig.add_subplot(223)
        ax3.imshow(X[0, :, :, 18, 0])
        ax3.set_title(name)

        removeThisValueFromHistogramToSeeBetter = 0
        _X = X[~(X == removeThisValueFromHistogramToSeeBetter)]
        ax4 = fig.add_subplot(224)
        ax4.hist(_X, 300)
        # ax4.set_title(name)
        ax4.set_title(' median= %.4f' % np.median(_X) + ' mean= %.4f' % np.mean(_X) + ' max= %.4f' % np.max(_X) + ' min= %.4f' % np.min(_X))

        print('median ' + name + ' = %.4f' % np.median(_X))
        print('mean  ' + name + ' = %.4f' % np.mean(_X))
        print('max  ' + name + ' = %.4f' % np.max(_X))
        print('min  ' + name + ' = %.4f' % np.min(_X))

        # ax2 = fig.add_subplot(232)
        # # ax2.imshow(X[i,:, :, 6, 0])
        # ax2.imshow(X[0, :, :, 6, 0])
        #
        # plt.imshow(X[:, :, 20, 0])
        # plt.show()

        print(imageNumber)
        save_path = savePathDataGeneration + '/' + str(imageNumber) + '.png'
        print(save_path)
        # plt.show()
        fig.savefig(save_path)
        plt.close(fig)

    def __data_generation(self, list_IDs_temp):
    # def test(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_classes))

        Xbefore = np.empty((self.batch_size, *self.dim, self.n_channels))
        Ybefore = np.empty((self.batch_size, *self.dim, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # X: we need to add a new axis for the channel dimension
            # Y: we need to transform to categorical

            # X[i,] = self._load3DImages(self.imagePathFolder, ID)[..., np.newaxis]
            if self.imageType == '.dcm':
                ID = ID[:-4]
                X[i,] = self._load3DImagesAndCrop(self.imagePathFolder, ID)[..., np.newaxis]
            else:
                X[i,] = self.load3DImagesNii(self.imagePathFolder, ID, self.imageType, self.crop_parameters)[..., np.newaxis]


            # add noise
            if (self.normalization_args['addNoise'] == 1):
                # print('add noise')
                X[i,] = addNoise(X[i,], self.normalization_args['normalization_threshold'], self.normalization_args['meanNoiseDistribution'], self.normalization_args['noiseMultiplicationFactor'])
                # X[i,] = addNoise(X[i,], mask, self.normalization_args['meanNoiseDistribution'], self.normalization_args['noiseMultiplicationFactor'])

            # normalization
            if (self.normalization_args['normalize'] == 1):
                # print('normalize')
               X[i,] = normalize(X[i,], self.normalization_args['normalization_threshold'])

            # histogram normalization
            if (self.normalization_args['histogramNormalize'] == 1):
                # print('normalize')

                # oldX = X

                X[i,] = histogramNormalize(X[i,], self.normalization_args['histogramNormalize_underThreshold'], self.normalization_args['histogramNormalize_strokeThreshold'], self.normalization_args['histogramNormalize_upperThreshold'])

                if (self.normalization_args['removeSignalunderRelativeThreshold']):
                    X[i,] = removeSignalunderRelativeThreshold(X[i,], self.normalization_args[
                        'removeSignalunderRelativeThreshold_cutOff'])

                # X[i,] = removeSignalunderRelativeThreshold(X[i,], relativeThreshold)
                # X[i,] = removeSignalunderRelativeThreshold(X[i,], 8)

                # import matplotlib.pyplot as plt
                #
                # fig = plt.figure(figsize=(14, 10))
                # ax1 = fig.add_subplot(231)
                # ax1.imshow(oldX[i,:, :, 6, 0])
                #
                # ax2 = fig.add_subplot(232)
                # ax2.imshow(X[i,:, :, 6, 0])

            # simpleNormalization
            if (self.normalization_args['simpleNormalize'] == 1):
                # oldX = X
                X[i,] = simpleNormalization(X[i,])


            # gaussian filter
            if (self.normalization_args['gaussian_filter'] == 1):
                # print('gaussian_filter')
                X[i,] = gaussian_smoothing(X[i,], self.normalization_args['gaussian_filter_hsize'], self.normalization_args['gaussian_filter_sigma'])


            # augmentation
            if (self.augment == 1):

                # prepare augmentation parameters using a random seed
                seed = []
                for x in range(0, 14):
                    seed.append(random.randint(0, 100)*random.randint(0, 1)/100)

                # limit the number of augmentation transformation to args.maxNumberOfTransformation
                keep = [0] * len(seed)
                # flip does not count in args.maxNumberOfTransformation
                keep[0] = 1


                for j in range(0,self.augmentation_args['maxNumberOfTransformation']):
                    keep[random.randint(0, len(seed)-1)] = 1

                seed = np.multiply(seed,keep)

                augmentation_parameters = {
                    'flip': self.augmentation_args['flip'] * seed[0],
                    'rotationAngle_Xaxis': self.augmentation_args['rotationRangeXAxis'] * seed[1],
                    'rotationAngle_Yaxis': self.augmentation_args['rotationRangeYAxis'] * seed[2],
                    'rotationAngle_Zaxis': self.augmentation_args['rotationRangeZAxis'] * seed[3],
                    'shift_Xaxis': self.augmentation_args['shiftXAxisRange'] * seed[4],
                    'shift_Yaxis': self.augmentation_args['shiftYAxisRange'] * seed[5],
                    'shift_Zaxis': self.augmentation_args['shiftZAxisRange'] * seed[6],
                    'zoom': 1 + [-1, 1][random.randrange(2)] * self.augmentation_args['zoomRange'] * seed[7],
                    'stretch_Xaxis': 1 + [-1, 1][random.randrange(2)] * self.augmentation_args['stretchFactorXAxisRange'] * seed[8],
                    'stretch_Yaxis': 1 + [-1, 1][random.randrange(2)] * self.augmentation_args['stretchFactorYAxisRange'] * seed[9],
                    'stretch_Zaxis': 1 + [-1, 1][random.randrange(2)] * self.augmentation_args['stretchFactorZAxisRange'] * seed[10],
                    'shear_NormalXAxis': [-1, 1][random.randrange(2)] * self.augmentation_args['shear_NormalXAxisRange'] * seed[11],
                    'shear_NormalYAxis': [-1, 1][random.randrange(2)] * self.augmentation_args['shear_NormalYAxisRange'] * seed[12],
                    'shear_NormalZAxis': [-1, 1][random.randrange(2)] * self.augmentation_args['shear_NormalZAxisRange'] * seed[13]}


                # augment
                X[i,] = augment(X[i,], **augmentation_parameters)
                Y[i,] = augment(Y[i,], **augmentation_parameters)

            # if self.intensity_augment == 1:

            #     try: 
            #         seg = nib.load(self.segPathFolder + ID)

            #one-hot encoding
            if self.labelType == '.dcm':
                Y[i,] = keras.utils.to_categorical(self._load3DImagesAndCrop(self.labelPathFolder, ID))
            else:
                Y[i,] = keras.utils.to_categorical(
                    self.load3DImagesNii(self.labelPathFolder, ID, self.labelType, self.crop_parameters),
                    num_classes=self.n_classes)


        return X.astype(self.variableType), Y.astype(self.variableType)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
          
        return X, Y

    def _getListOfSlicesNumbers(self, pathToDicomFolder, caseNumber):

        # returns the list of slice numbers for a given caseNumber in the folder pathToDicomFolder
        # the files in the Dicom folder must be saved in the form 00100001_001.dcm
        # with 00100001 the caseNumber and 001 the sliceNumber

        # listOfPathToDicoms = glob.glob(pathToDicomFolder + '/' + caseNumber + '*')
        #
        # # use the property that set have unique elements
        # caseNumbers = set()
        # for i in range(len(listOfPathToDicoms)):
        #     caseNumbers.add(listOfPathToDicoms[i][-7:-4])
        #
        # return sorted(list(caseNumbers))
        #
        # to increase spead if DB is large
        return ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040']#, '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181']



    def _load3DImages(self, pathToDicomFolder,caseNumber):

        # this function loads the dicoms from a Folder and save it in a 3D array: x, y, z
        # the files in the Dicom array must be saved in the form 00100001_001.dcm
        # with 00100001 the caseNumber and 001 the sliceNumber

        _listOfSlicesNumbers = self._getListOfSlicesNumbers(pathToDicomFolder, caseNumber)

        # get the first dicomImage and read the size in x and y direction
        pathToFirstDicom = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[1] + self.imageType
        ds_firstDicom = pydicom.dcmread(pathToFirstDicom)
        pix = ds_firstDicom.pixel_array

        # initialize image_array to the right size
        image_array = np.zeros((pix.shape[0], pix.shape[1], len(_listOfSlicesNumbers)))

        for j in range(len(_listOfSlicesNumbers)):

            # pathToDicom = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[j] + self.imageType
            # ds = pydicom.dcmread(pathToDicom)
            # pix = ds.pixel_array
            # image_array[:, :, j] = pix

            pathToImage = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[j] + self.imageType

            if self.imageType == '.dcm':
                ds = pydicom.dcmread(pathToImage)
                pix = ds.pixel_array
                image_array[:, :, j] = pix

            else:
                image_array[:, :, j] = Image.open(pathToImage)

        return image_array


    def _load3DImagesAndCrop(self, pathToDicomFolder, caseNumber):

        # this function loads the dicoms from a Folder and save it in a 3D array: x, y, z
        # the files in the Dicom array must be saved in the form 00100001_001.dcm
        # with 00100001 the caseNumber and 001 the sliceNumber

        _listOfSlicesNumbers = self._getListOfSlicesNumbers(pathToDicomFolder, caseNumber)

        # get the first dicomImage and read the size in x and y direction
        pathToFirstDicom = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[1] + self.imageType
        ds_firstDicom = pydicom.dcmread(pathToFirstDicom)
        pix = ds_firstDicom.pixel_array

        # initialize image_array to the right size
        image_array = np.zeros((pix.shape[0], pix.shape[1], len(_listOfSlicesNumbers)))

        for j in range(len(_listOfSlicesNumbers)):

            # pathToDicom = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[j] + self.imageType
            # ds = pydicom.dcmread(pathToDicom)
            # pix = ds.pixel_array
            # image_array[:, :, j] = pix

            pathToImage = pathToDicomFolder + '/' + caseNumber + '_' + _listOfSlicesNumbers[j] + self.imageType

            if self.imageType == '.dcm':
                ds = pydicom.dcmread(pathToImage)
                pix = ds.pixel_array
                image_array[:, :, j] = pix

            else:
                image_array[:, :, j] = Image.open(pathToImage)


        # we crop down to 96 (26:122) x 80 (25:105)
        #  this permit a network depth of 3

        return image_array[43:139,36:164,48:112]

        # timestamps for debugging
        # t1 = datetime.datetime.now()
        # t2 = datetime.datetime.now()
        # delta=t2-t1
        # print(delta.microseconds)


    def load3DImagesNii(self, pathToNiiFolder, caseNumber, imageType, crop_parameters):

        # this function loads the .nii.gz from a Folder and save it in a 3D array: x, y, z
        # the files in the Nifti array must be saved in the form 100_00100904_0.nii.gz
        # with 100_00100904_0 the caseNumber
        
        pathToNiiGz = pathToNiiFolder + '/' + caseNumber + imageType
        
        image_array = np.asanyarray(nib.load(pathToNiiGz).dataobj).squeeze()

        image_array = np.round(image_array).astype(np.int16)
        
        return image_array[crop_parameters[0]: crop_parameters[1], crop_parameters[2]: crop_parameters[3], crop_parameters[4]: crop_parameters[5]]
