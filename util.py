import os
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

from tensorflow.keras import backend as K

from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import (load_img, img_to_array)
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2

import errno
#from importDicoms import *
import scipy.stats
from scipy import signal

from skimage.measure import label
from scipy.stats import mode
import nibabel as nib


def _get_dim_ordering():
#    global row_axis
#    global col_axis
#    global channel_axis

    #if K.image_dim_ordering() == 'tf':
    if K.image_data_format() == 'channels_last':
        row_axis = 1
        col_axis = 2
        channel_axis = 3
    else:
        channel_axis = 1
        row_axis = 2
        col_axis = 3

    return row_axis, col_axis, channel_axis


def _get_channel_axis():
    _, _, channel_axis = _get_dim_ordering()
    return channel_axis


def _get_row_axis():
    row_axis, _, _ = _get_dim_ordering()
    return row_axis


def _get_col_axis():
    _, col_axis, _ = _get_dim_ordering()
    return col_axis


def save_model(model, save_path, model_name='model'):
    # save model structure
    model_path = os.path.join(save_path, "{}.json".format(model_name))
    f = open(model_path, 'w')
    model_json = model.to_json()
    f.write(model_json)
    f.close

    img_path = os.path.join(save_path, "{}.png".format(model_name))
    # vis_util.plot(model, to_file=img_path, show_shapes=True)
    plot_model(model, to_file=img_path, show_shapes=True)


# Build absolute image paths
def absolute_paths(basenames, img_root, mask_root):
#https://github.com/nicolov/segmentation_keras/blob/master/train.py
    img_fnames = [os.path.join(img_root, f) for f in basenames]
    mask_fnames = [os.path.join(mask_root, f) for f in basenames]
    return img_fnames, mask_fnames


def load_img_array(fname, grayscale=False, target_size=None):
    """Loads and image file and returns an array."""
    img = load_img(fname,
                   grayscale=grayscale,
                   target_size=target_size)
    x = img_to_array(img)
    return x


def mask2onehot(masks, nb_labels=2, **kwargs):
    #print('{}x{}'.format(masks.shape[0], masks.shape[1]))
    #print('{}x{}x{}'.format(masks.shape[0], masks.shape[1], masks.shape[2]))
    masks = np.squeeze(masks, axis=(2,))
    # width, height = masks.shape[0], masks.shape[1]
    width, height = masks.shape[1], masks.shape[0]
    #print(masks.shape)
    #print('{}x{}'.format(masks.shape[0], masks.shape[1]))
    masks_cat = np.zeros((height, width, nb_labels))
    for c in range(nb_labels):
        mask = (masks == c).astype(int)
        #print(mask.shape)
        #print(masks_cat[:, :, c].shape)
        masks_cat[:, :, c] = mask
    #masks_cat = np.reshape(masks_cat, (width * height, nb_labels))
    #print(masks_cat.shape)
    return masks_cat

#
# def normalize(x, epsilon=1e-7, axis=(1,2), **kwargs):
# #def normalize(x, epsilon=1e-7, axis=(0,1), **kwargs):
#
#     x -= np.mean(x, axis=axis, keepdims=True)
#     x /= np.std(x, axis=axis, keepdims=True) + epsilon
#
#     return x

#
# def cropAndNormalize(img):
#
#     # print(img.shape)
#     import matplotlib.pyplot as plt
#     # plt.imshow(img)
#     # plt.show()
#
#     img = img[38:102, 34:98]
#     # print(img.shape)
#
#     # print(np.mean(img))
#     # print(np.std(img))
#     #
#     # plt.imshow(img)
#     # plt.show()
#     epsilon = 1e-7
#     # plt.imshow((img - np.mean(img))/(np.std(img)+epsilon))
#
#     # normalize
#     # img_normalized =  normalize_im_single(img)
#     img_normalized =  (img - np.mean(img))/(np.std(img)+epsilon)
#     #
#     # print(img)
#     # print(img_normalized)
#     # plt.imshow(img_normalized)
#     # plt.show()
#
#     # plt.imshow(img - np.std(img))
#     # plt.show()
#
#     return img_normalized
#     # return img
def crop(img, **kwargs):

    # print(img.shape)
    import matplotlib.pyplot as plt

    img = img[38:102, 34:98]

    # plt.imshow(np.squeeze(img))
    # plt.show()
    # print(img.shape)

    # print(np.mean(img))
    # print(np.std(img))
    #
    # plt.imshow(img)
    # plt.show()

    return img

def normalize_ims(x, epsilon=1e-7, axis=(0,1), **kwargs):
# mean = K.mean(x, axis=all_dims, keepdims=True)
# std = K.std(x, axis=all_dims, keepdims=True)
# x = (x - mean) / (std + epsilon)

    x_cpy = np.float32(x.copy())
    mean  = np.mean(x_cpy, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True) + epsilon
    x_norm = np.divide((x_cpy-mean), std)
    return x_norm, mean, std

def normalize_im_single(im, epsilon=1e-7, **kwargs):
    '''
    make image zero mean and unit standard deviation
    '''

    # img_o = np.float32(im.copy())
    img_o = im

    m = np.mean(img_o)
    # print(m)
    s = np.std(img_o) + epsilon
    return np.divide((img_o - m), s)

    # return np.divide( img_o,s)

def clahe_im(im, **kwargs):
    im = np.array(im * 255, dtype=np.uint8)
    clahe = cv2.createCLAHE(tileGridSize=(1, 1))
    res = clahe.apply(im)
    res = np.expand_dims(res, axis=3)
    return res

def plot_learning_curve(history, outdir):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(outdir, "accuracy_curve.png"))

    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(outdir, "loss_curve.png"))

def _get_strides_factor(target_shape, reference_shape):
    row_axis, col_axis, channel_axis = _get_dim_ordering()

    stride_height = int(round(target_shape[row_axis] / reference_shape[row_axis]))
    stride_width = int(round(target_shape[col_axis] / reference_shape[col_axis]))

    return stride_height, stride_width

def _get_tensor_shape(x):
    row_axis, col_axis, channel_axis = _get_dim_ordering()
    tensor_shape = K.int_shape(x)

    row_dim = tensor_shape[row_axis]
    col_dim = tensor_shape[col_axis]
    ch_dim = tensor_shape[channel_axis]

    return row_dim, col_dim, ch_dim

def get_trained_model(model, args):
    """ Returns a model with loaded weights. """


    def load_tf_weights():
        """ Load pretrained weights converted from Caffe to TF. """

        # 'latin1' enables loading .npy files created with python2
        weights_data = np.load(args.weights_path, encoding='latin1').item()

        for layer in model.layers:
            if layer.name in weights_data.keys():
                layer_weights = weights_data[layer.name]
                layer.set_weights((layer_weights['weights'],
                                   layer_weights['biases']))

    def load_keras_weights():
        """ Load a Keras checkpoint. """
        model.load_weights(args.weights_path)

    if args.weights_path.endswith('.npy'):
        load_tf_weights()
    elif args.weights_path.endswith('.hdf5'):
        load_keras_weights()
    else:
        raise Exception("Unknown weights format.")

    model.summary()

    return model

def create_out_dir(model_date, args):

    outims = os.path.join(args.datatest, 'preds/' + args.model_name + '/' + model_date + args.outims)

    print('outims')
    print(outims)


    if os.path.exists(outims) is False:
        os.makedirs(outims)

    outgts = os.path.join(args.datatest, 'preds/' + args.model_name + '/' + model_date + args.outgts)
    if os.path.exists(outgts) is False:
        os.makedirs(outgts)

    outfig = os.path.join(args.datatest, 'preds/' + args.model_name + '/' + model_date + args.outfig)
    if os.path.exists(outfig) is False:
        os.makedirs(outfig)

    outpred = os.path.join(args.datatest, 'preds/' + args.model_name + '/' + model_date + args.outpred)
    if os.path.exists(outpred) is False:
        os.makedirs(outpred)

    outctr = os.path.join(args.datatest, 'preds/' + args.model_name + '/' + model_date + args.outctr)
    if os.path.exists(outctr) is False:
        os.makedirs(outctr)

    outover = os.path.join(args.datatest, 'preds/' + args.model_name + '/' + model_date + args.outover)
    if os.path.exists(outover) is False:
        os.makedirs(outover)

    outovergt = os.path.join(outover, 'gts/')
    if os.path.exists(outovergt) is False:
        os.makedirs(outovergt)

    outoverpreds = os.path.join(outover, 'preds/')
    if os.path.exists(outoverpreds) is False:
        os.makedirs(outoverpreds)

    outstats = os.path.join(args.datatest, 'preds/' + args.model_name + '/' + model_date + args.outstats)
    if os.path.exists(outstats) is False:
        os.makedirs(outstats)

    outStrokeSummary = os.path.join(args.datatest, 'preds/' + args.model_name + '/' + model_date + args.outStrokeSummary)
    if os.path.exists(outStrokeSummary) is False:
        os.makedirs(outStrokeSummary)

    return outims, outgts, outfig, outpred, outctr, outover, outstats, outStrokeSummary

def get_contours(mask):
    mask = np.where(mask > 0, 255, 0).astype('uint8')
    im2, coords, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not coords:
        print("No contour detected.")
        coords = np.ones((1, 1, 1, 2), dtype='int')
    if len(coords) > 1:
        print("Multiple contours detected.")
        lengths = [len(coord) for coord in coords]
        coords = [coords[np.argmax(lengths)]]

    coords = np.squeeze(coords[0], axis=(1,))
    coords = np.append(coords, coords[:1], axis=0)

    return coords

def contour_for_vis(contours, image_shape=(256, 256)):
#https://stackoverflow.com/questions/40335161/how-to-fill-numpy-array-of-zeros-with-ones-given-indices-coordinates
    # arr = np.zeros(image_shape)
    #
    # points = np.indices(arr.shape).reshape(2, -1).T
    # path = Path(contours)
    # mask = path.contains_points(points, radius=1e-9)
    # mask = mask.reshape(arr.shape).astype(arr.dtype)
    #
    # return mask

    msk = np.zeros(image_shape)
    row_indices = contours[:, 1]
    col_indices = contours[:, 0]
    msk[row_indices, col_indices] = 1

    return msk

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts



def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def generatePartitionTrainAndValidFromFolder(imagePathFolder,_validProportion,shuffle_train_val,image_type='.dcm'):
    # generate List of train and valid IDs from the DicomFolderPaths randomly

    _listOfCases = listOfCasesInFolder(imagePathFolder, image_type)

    # _listOfSlicesNumbers = listOfSlicesNumbersInDicomFolder(imagePathFolder)
    
    # randomIndex = np.arange(len(_listOfCases))
    index = np.arange(len(_listOfCases))

    if shuffle_train_val:
        np.random.shuffle(index)

    indexTrain = index[0:(len(_listOfCases) - int(np.floor(len(_listOfCases) * _validProportion)))]
    indexValid = index[(len(_listOfCases) - int(np.floor(len(_listOfCases) * _validProportion))):]

    _listOfTrainCasesID = [_listOfCases[k] for k in indexTrain]
    _listOfValidCasesID = [_listOfCases[k] for k in indexValid]

    partition = {'train': _listOfTrainCasesID, 'validation': _listOfValidCasesID}

    return partition


def removeStrokeBelowThreshold(listOfCases, labelsPathFolder, image_type='.dcm', threshold=20):

    valid = []
    for case in listOfCases:
        gts = nib.load(labelsPathFolder + '/' + case + image_type).get_fdata()

        if threshold >= 0 and np.sum(gts) >= threshold:# or np.sum(gts) < -1. * threshold:
            valid.append(case)
        elif threshold < 0 and np.sum(gts) < -threshold:
            valid.append(case)

    return valid

def generatePartitionTrainAndValidFromFolderRandomly(imagePathFolder, _validProportion, shuffle_train_val, image_type='.dcm', labelPathFolder=None, label_type='.nii', threshold=20):
    # generate List of train and valid IDs from the DicomFolderPaths randomly

    _listOfCases = listOfCasesInFolder(imagePathFolder, image_type)

    if labelPathFolder is not None:
        _listOfCases = removeStrokeBelowThreshold(_listOfCases, labelPathFolder, image_type=label_type, threshold=threshold)

    # _listOfSlicesNumbers = listOfSlicesNumbersInDicomFolder(imagePathFolder)
    
    # randomIndex = np.arange(len(_listOfCases))
    index = np.arange(len(_listOfCases))

    if shuffle_train_val:
        np.random.shuffle(index)

    validNumbers = int(np.floor(len(_listOfCases) * _validProportion))

    indexValid = random.sample(range(0, len(_listOfCases)), validNumbers)
    # indexValid = [random.randrange(0, len(_listOfCases)) for iter in range(validNumbers)]
    indexTrain = []
    for k in index:
        if k not in indexValid:
            indexTrain.append(k)

    _listOfTrainCasesID = [_listOfCases[k] for k in indexTrain]
    _listOfValidCasesID = [_listOfCases[k] for k in indexValid]

    partition = {'train': _listOfTrainCasesID, 'validation': _listOfValidCasesID}

    return partition

def chunks(list_of_cases, k):
    for i in range(0, len(list_of_cases), k):
        yield list_of_cases[i:i+k]

def generatePartitionCrossValidation(imagePathFolder, k, shuffle_train, labelPathFolder=None, image_type='.dcm', label_type='.nii', threshold=20):
    list_of_cases = listOfCasesInFolder(imagePathFolder, image_type)
    
    if labelPathFolder is not None:
        list_of_cases = removeStrokeBelowThreshold(list_of_cases, labelPathFolder, image_type=label_type, threshold=threshold)

    n = int(np.ceil(len(list_of_cases)/k))
    
    if shuffle_train:
        np.random.shuffle(list_of_cases)

    return list(chunks(list_of_cases, n))

def listOfCasesInFolder(pathToFolder, image_type='.dcm'):
    listOfImages = []
    listOfFiles = os.listdir(pathToFolder)
    for f in listOfFiles:
        if f.endswith(image_type) and f[0] != '.':
            listOfImages.append(f.split('.')[0])
    
    return list(listOfImages)

# not sure if this always works
def getLargestConnectedComponent(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    largest=max(list_seg, key=lambda x:x[1])[0]
    labels_max=(labels == largest).astype(int)
    return labels_max

def addNoise(X, threshold, meanNoiseDistribution, noiseMultiplicationFactor):
    # the stddev of the noise distribution is equal to the stddev of the voxel above the threshold of the image multiplied with the noiseMultiplicationFactor

    mask = X > threshold
    mask.astype(np.int)

    # iterate over channels
    for channel in range(mask.shape[3]):
        mask[:,:,:,channel] = getLargestConnectedComponent(mask[:,:,:,channel])

    X = np.multiply(X,mask)

    factor = np.random.uniform(0, noiseMultiplicationFactor)

    stddevNoiseDistribution = factor * np.std(X[X > threshold])

    # where loc is the mean and scale is the std dev
    noiseToAdd = scipy.stats.norm.rvs(loc=meanNoiseDistribution, scale=stddevNoiseDistribution, size = X.shape)

    return np.multiply(np.add(X,noiseToAdd), mask)

def normalize(X, threshold):

    # normalizedX = np.divide((X - np.mean(X[X > threshold])), np.std(X[X > threshold]))

    mask = X > threshold
    mask.astype(np.int)

    # iterate over channels
    for channel in range(mask.shape[3]):
        mask[:, :, :, channel] = getLargestConnectedComponent(mask[:, :, :, channel])

    X = np.multiply(X, mask)

    normalizedX = np.divide((X - np.mean(X[X > threshold])), np.std(X[X > threshold]))

    # normalizedX = gaussian_smoothing(normalizedX, hsize=3, sigma=1)

    return np.multiply(normalizedX, mask)

def gaussian_smoothing(X, hsize, sigma):

    # first build the smoothing kernel
    # sigma = 1.0  # width of kernel

    x = np.arange(-3, 4, 1)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3, 4, 1)
    z = np.arange(-3, 4, 1)

    x = np.arange(-hsize+1, hsize, 1)  # coordinate arrays -- make sure they contain 0!
    y = np.arange(-hsize+1, hsize, 1)
    z = np.arange(-hsize+1, hsize, 1)

    xx, yy, zz = np.meshgrid(x, y, z)
    kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))

    # iterate over channels
    for channel in range(X.shape[3]):
        X[:, :, :, channel] = signal.convolve(X[:, :, :, channel], kernel, mode="same")

    return X

def histogramNormalize(X, underThreshold,strokeThreshold,upperThreshold):
    
    # underThreshold = 40
    # strokeThreshold = 200
    # upperThreshold = 500

    # mask
    mask = X > underThreshold
    mask.astype(np.int)

    # upper threshold
    X = np.multiply(X.clip(0, upperThreshold), mask)

    # find the non stroke side
    mask_aboveStrokeThreshold = X > strokeThreshold

    # X_left_hemisphere = X[:, 0:64, :, :]
    # X_right_hemisphere = X[:, 64:128, :]
    #
    # leftCount = mask_aboveStrokeThreshold[:, 0:64, :, :].sum()
    # rightCount = mask_aboveStrokeThreshold[:, 64:128, :, :].sum()

    X_left_hemisphere = X[:, 0:40, :, :]
    X_right_hemisphere = X[:, 40:80, :, :]

    leftCount = mask_aboveStrokeThreshold[:, 0:40, :, :].sum()
    rightCount = mask_aboveStrokeThreshold[:, 40:80, :, :].sum()

    if (leftCount >= rightCount):
        normalizationMean = np.mean(X_right_hemisphere[np.nonzero(X_right_hemisphere)])
    else:
        normalizationMean = np.mean(X_left_hemisphere[np.nonzero(X_left_hemisphere)])

    # divide by mean
    X = np.divide(X, normalizationMean)

    # rescale
    X_rescaled = np.divide(np.subtract(X, X.min()), np.subtract(X.max(), X.min()))

    # substract mode
    X_rescaled = np.subtract(X_rescaled, mode(X_rescaled[np.nonzero(X_rescaled)], axis=None)[0][0])

    # try multiply mask with 100
    mask = np.multiply(mask, 100)

    # mask
    return np.multiply(X_rescaled, mask)

def removeSignalunderRelativeThreshold(X, relativeThreshold):
    # relativeThreshold in percent above the mean signal of the "most normal" brain side

    # relativeThreshold = 10 means 10% above the mean signal of the "most normal" brain side

    mask = X > relativeThreshold

    return np.multiply(X.clip(0), mask)

def simpleNormalization(array):
    vol = array

    if np.amax(vol) > 255:
        vol[vol < 30] = 0
    else:
        vol -= np.max(vol[0:20,:,-1])

    vol = np.clip(vol,0,np.percentile(vol,99.5))
    # vol /= np.max(vol)

    vol -= np.mean(vol)
    vol /= np.std(vol)
    vol = -1 + 2 * (vol - np.min(vol)) / (np.max(vol) - np.min(vol))

    return vol

def CTNormalization(array):
    vol = np.clip(array, 0.001, 100)
    
    # vol -= np.mean(vol)
    # vol /= np.std(vol)
    vol = -1 + 2 * (vol - np.min(vol)) / (np.max(vol) - np.min(vol))

    return vol

#1909.02642
def intensityNormalization(img, N=50, augment=True):

    #normalize
    aug = np.clip(img, 30, np.percentile(img, 99.5)).astype(int) - 30
    
    #augment
    if augment:
        random = np.random.uniform(np.min(aug), (np.max(aug)), size=(N + int(np.max(aug))))
        
        #moving average
        random = np.convolve(random, np.ones((N,))/N, mode='same')
        random = random[N//2:-N//2]

        #linear component
        lin = np.arange(int(np.max(aug)))
        sgn = np.random.uniform(-1,1)
        lin =  0.5 * sgn * (lin - lin[-1]/2)
        
        #add components and rescale
        prox = random + lin
        prox = np.max(aug) * (prox - np.min(prox)) / (np.max(prox) - np.min(prox))
        
        #augmentation
        for i, val in enumerate(np.arange(np.max(aug))):
            aug[aug == val] = prox[i]
    
    #rescale to [-1,1] 
    aug = -1 + 2 * (aug - np.min(aug)) / (np.max(aug) - np.min(aug)) 

    return aug

    
def mosaic(Image3D,numberOfRows=5):

    numberOfImages = Image3D.shape[2]
    img_list = [Image3D[:, :, i] for i in range(numberOfImages)]
    numberOfColumns = int(len(img_list) / numberOfRows)

    i = 0
    for r in range(numberOfRows):
        for c in range(numberOfColumns):
            if c == 0:
                row = img_list[i]
            else:
                row = np.append(row, img_list[i], axis=1)
            i = i + 1

        if r == 0:
            mosaic = row
        else:
            mosaic = np.append(mosaic, row, axis=0)

    return mosaic
