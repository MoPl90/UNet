from keras import backend as K

import numpy as np
import scipy.ndimage as ndi


# range recommendation

# rotationAngle_Xaxis: +/- 2
# shift_Xaxis = +/- 0.02
# zoom   0.95 - 1.05
# stretchFactor  0.95 - 1.05
# shearFactor  +/- 0.03

def augment(matrix, flip = 0, rotationAngle_Xaxis = 0, rotationAngle_Yaxis = 0, rotationAngle_Zaxis = 0, shift_Xaxis = 0, shift_Yaxis = 0, shift_Zaxis = 0, zoom = 1,
            stretch_Xaxis = 0, stretch_Yaxis = 0, stretch_Zaxis = 0, shear_NormalXAxis = 0, shear_NormalYAxis = 0, shear_NormalZAxis = 0):



    # print("augment")
    if (flip != 0):
        matrix = flip_3D(matrix)

    if (rotationAngle_Xaxis != 0):
        matrix = rotation_3D(matrix, rotationAngle_Xaxis, rotation_axis='x')

    if (rotationAngle_Yaxis != 0):
        matrix = rotation_3D(matrix, rotationAngle_Yaxis, rotation_axis='y')

    if (rotationAngle_Zaxis != 0):
        matrix = rotation_3D(matrix, rotationAngle_Zaxis, rotation_axis='z')

    if (shift_Xaxis != 0):
        matrix = shift_3D(matrix, shift_Xaxis, shift_axis='x')

    if (shift_Yaxis != 0):
        matrix = shift_3D(matrix, shift_Yaxis, shift_axis='y')

    if (shift_Zaxis != 0):
            matrix = shift_3D(matrix, shift_Zaxis, shift_axis='z')

    if (zoom != 1):
        matrix = zoom_3D(matrix, zoom)

    if (stretch_Xaxis != 1):
        matrix = stretch_3D(matrix, stretch_Xaxis, stretch_axis='x')

    if (stretch_Yaxis != 1):
        matrix = stretch_3D(matrix, stretch_Yaxis, stretch_axis='y')

    if (stretch_Zaxis != 1):
        matrix = stretch_3D(matrix, stretch_Zaxis, stretch_axis='z')

    if (shear_NormalXAxis != 0):
        matrix = shear_3D(matrix, shear_NormalXAxis, shearNormalAxis='x')

    if (shear_NormalYAxis != 0):
        matrix = shear_3D(matrix, shear_NormalYAxis, shearNormalAxis='y')

    if (shear_NormalZAxis != 0):
        matrix = shear_3D(matrix, shear_NormalZAxis, shearNormalAxis='z')



    return matrix





def transform_matrix_offset_center_3D(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5

    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)

    return transform_matrix


def apply_transform_3D(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):

    x = np.rollaxis(x, channel_index, 0)

    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, 3]

    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]

    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)

    return x




def rotation_3D(inputMatrix, rotationAngle, row_index=0, col_index=1, depth_index=2, channel_index=3, rotation_axis='z',
                fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * rotationAngle

    if (rotation_axis == 'z'):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                                    [np.sin(theta), np.cos(theta), 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

    if (rotation_axis == 'y'):
        rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                                    [0, 1, 0, 0],
                                    [-np.sin(theta), 0, np.cos(theta), 0],
                                    [0, 0, 0, 1]])

    if (rotation_axis == 'x'):
        rotation_matrix = np.array([[1, 0, 0, 0],
                                    [0, np.cos(theta), -np.sin(theta), 0],
                                    [0, np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 0, 1]])

    h, w, d = inputMatrix.shape[row_index], inputMatrix.shape[col_index], inputMatrix.shape[depth_index]
    transform_matrix = transform_matrix_offset_center_3D(rotation_matrix, h, w, d)
    rotatedMatrix = apply_transform_3D(inputMatrix, transform_matrix, channel_index, fill_mode, cval)

    return rotatedMatrix



def shift_3D(inputMatrix, shift, row_index=0, col_index=1, depth_index=2, channel_index=3, shift_axis='z',
    fill_mode='nearest', cval=0.):

    # shift is multiplication factor of the size of the image

    if (shift_axis == 'z'):
        t = inputMatrix.shape[depth_index] * shift

        translation_matrix = np.array([ [1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, t],
                                        [0, 0, 0, 1]])

    if (shift_axis == 'y'):
        t = inputMatrix.shape[col_index] * shift
        translation_matrix = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, t],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])

    if (shift_axis == 'x'):
        t = inputMatrix.shape[row_index] * shift
        translation_matrix = np.array([[1, 0, 0, t],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])

    # transform_matrix = translation_matrix  # no need to do offset
    # rotatedMatrix = apply_transform(inputMatrix, transform_matrix, channel_index, fill_mode, cval)
    shiftedMatrix = apply_transform_3D(inputMatrix, translation_matrix, channel_index, fill_mode, cval)
    return shiftedMatrix




def zoom_3D(inputMatrix, zoom, row_index=0, col_index=1, depth_index=2, channel_index=3,
             fill_mode='nearest', cval=0.):

    # zoom factor will be multiplied to the current volume


    zoom_matrix = np.array([[zoom, 0, 0, 0],
                            [0, zoom, 0, 0],
                            [0, 0, zoom, 0],
                            [0, 0, 0, 1]])

    h, w, d = inputMatrix.shape[row_index], inputMatrix.shape[col_index], inputMatrix.shape[depth_index]
    transform_matrix = transform_matrix_offset_center_3D(zoom_matrix, h, w, d)
    zoomedMatrix = apply_transform_3D(inputMatrix, transform_matrix, channel_index, fill_mode, cval)

    return zoomedMatrix



def stretch_3D(inputMatrix, stretchFactor, row_index=0, col_index=1, depth_index=2, channel_index=3, stretch_axis='z',
             fill_mode='nearest', cval=0.):

    # stretchFactor  will be multiplied in the given direction

    if (stretch_axis == 'z'):
        stretch_matrix = np.array([ [1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, stretchFactor, 0],
                                        [0, 0, 0, 1]])

    if (stretch_axis == 'y'):
        stretch_matrix = np.array([[1, 0, 0, 0],
                                       [0, stretchFactor, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])

    if (stretch_axis == 'x'):
        stretch_matrix = np.array([[stretchFactor, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])


    h, w, d = inputMatrix.shape[row_index], inputMatrix.shape[col_index], inputMatrix.shape[depth_index]
    transform_matrix = transform_matrix_offset_center_3D(stretch_matrix, h, w, d)
    stretchedMatrix = apply_transform_3D(inputMatrix, transform_matrix, channel_index, fill_mode, cval)

    return stretchedMatrix



def shear_3D(inputMatrix, shearFactor, row_index=0, col_index=1, depth_index=2, channel_index=3, shearNormalAxis ='z',
                 fill_mode='nearest', cval=0.):

    if (shearNormalAxis == 'z'):
        shear_matrix = np.array([[1, shearFactor, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

    if (shearNormalAxis == 'x'):
        shear_matrix = np.array([[1, 0, 0, 0],
                                   [0, 1, shearFactor, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

    if (shearNormalAxis == 'y'):
        shear_matrix = np.array([[1, 0, shearFactor, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

    h, w, d = inputMatrix.shape[row_index], inputMatrix.shape[col_index], inputMatrix.shape[depth_index]
    transform_matrix = transform_matrix_offset_center_3D(shear_matrix, h, w, d)
    shearedMatrix = apply_transform_3D(inputMatrix, transform_matrix, channel_index, fill_mode, cval)

    return shearedMatrix



def flip_3D(inputMatrix, flipAxis = 1):
    return np.flip(inputMatrix, flipAxis)




# def addNoise_3D(inputMatrix, rotationAngle, row_index=0, col_index=1, depth_index=2, channel_index=3, rotation_axis='z',
#                 fill_mode='nearest', cval=0.):
#
#     if (rotation_axis == 'z'):
#         rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
#                                     [np.sin(theta), np.cos(theta), 0, 0],
#                                     [0, 0, 1, 0],
#                                     [0, 0, 0, 1]])
#
#     if (rotation_axis == 'y'):
#         rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta), 0],
#                                     [0, 1, 0, 0],
#                                     [-np.sin(theta), 0, np.cos(theta), 0],
#                                     [0, 0, 0, 1]])
#
#     if (rotation_axis == 'x'):
#         rotation_matrix = np.array([[1, 0, 0, 0],
#                                     [0, np.cos(theta), -np.sin(theta), 0],
#                                     [0, np.sin(theta), np.cos(theta), 0],
#                                     [0, 0, 0, 1]])
#
#     h, w, d = inputMatrix.shape[row_index], inputMatrix.shape[col_index], inputMatrix.shape[depth_index]
#     transform_matrix = transform_matrix_offset_center_3D(rotation_matrix, h, w, d)
#     rotatedMatrix = apply_transform_3D(inputMatrix, transform_matrix, channel_index, fill_mode, cval)
#
#     return rotatedMatrix




def random_rotation(x, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0.):

    # theta = np.pi / 180 * np.random.uniform(-rg, rg)
    theta = np.pi / 180 * 40

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]

    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)

    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x



def random_shift(x, wrg, hrg, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0.):
    h, w = x.shape[row_index], x.shape[col_index]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shear(x, intensity, row_index=1, col_index=2, channel_index=0,
                 fill_mode='nearest', cval=0.):
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x