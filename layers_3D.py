from tensorflow.keras import backend as K

from tensorflow.keras.layers import Lambda

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Reshape

from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import Cropping3D
from tensorflow.keras.layers import GlobalAveragePooling3D
from tensorflow.keras.layers import MaxPooling3D

from tensorflow.keras.regularizers import l2

from util import _get_dim_ordering, _get_channel_axis, _get_strides_factor, _get_tensor_shape

# from layers import *


def _bn_act(**layer_params):
    """
        Block: BN -> Activation(RELU)
        :param layer_params:
        :return:
    """

    channel_axis = _get_channel_axis()
    axis = layer_params.setdefault('axis', channel_axis)
    batch_norm = layer_params.setdefault('batch_norm', True)
    activation = layer_params.setdefault('activation', 'relu')

    def func(input_tensor):
        x = input_tensor
        if batch_norm:
            x = BatchNormalization(axis=axis)(x)
        act = Activation(activation=activation)(x)
        return act

    return func


def _conv_3D(**layer_params):
    """
        Block: Conv2D
        :param conv_params:
        :return:
    """
    filters = layer_params['filters']
    kernel_size = layer_params['kernel_size']
    strides = layer_params.setdefault('strides', (1, 1, 1))
    dilation_rate = layer_params.setdefault('dilation_rate', 1)
    kernel_initializer = layer_params.setdefault('kernel_initializer', 'he_normal')
    padding = layer_params.setdefault('padding', 'same')
    activation = layer_params.setdefault('activation', 'linear')

    # block_no = conv_params.setdefault('block_no', 1)
    # layer_no = conv_params.setdefault('layer_no', 1)
    # name = conv_params.setdefault('name', 'conv_{}_{}'.format(block_no, layer_no))

    def func(input_tensor):
        conv = Conv3D(
            # name=name,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer,
            activation=activation,
            use_bias=False)(input_tensor)

        return conv

    return func


def _deconv_3D(**layer_params):
    """
        Block: Conv2DTranspose
        :param conv_params:
        :return:
    """
    filters = layer_params['filters']
    kernel_size = layer_params['kernel_size']
    strides = layer_params.setdefault('strides', (1, 1, 1))
    dilation_rate = layer_params.setdefault('dilation_rate', 1)
    kernel_initializer = layer_params.setdefault('kernel_initializer', 'he_normal')
    padding = layer_params.setdefault('padding', 'same')
    activation = layer_params.setdefault('activation', 'linear')

    # block_no = conv_params.setdefault('block_no', 1)
    # layer_no = conv_params.setdefault('layer_no', 1)
    # name = conv_params.setdefault('name', 'conv_{}_{}'.format(block_no, layer_no))

    def func(input_tensor):
        conv = Conv3DTranspose(
            # name=name,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer,
            activation=activation,
            use_bias=False)(input_tensor)
        return conv

    return func


def _deconv_bn_act_3D(**layer_params):
    """
        Block: BN -> Activation(Relu) -> Conv
        :param conv_params:
        :return:
    """
    channel_axis = _get_channel_axis()
    filters = layer_params['filters']
    kernel_size = layer_params['kernel_size']
    strides = layer_params.setdefault('strides', (1, 1, 1))
    dilation_rate = layer_params.setdefault('dilation_rate', 1)
    kernel_initializer = layer_params.setdefault('kernel_initializer', 'he_normal')
    padding = layer_params.setdefault('padding', 'same')
    kernel_regularizer = layer_params.setdefault('kernel_regularizer', l2(1.e-4))
    axis = layer_params.setdefault('axis', channel_axis)
    activation = layer_params.setdefault('activaton', 'relu')
    batch_norm = layer_params.setdefault('batch_norm', True)



    def func(input_tensor):
        weights = _deconv_3D(filters=filters, kernel_size=kernel_size,
                             strides=strides, padding=padding,
                             dilation_rate=dilation_rate,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer)(input_tensor)
        act = _bn_act(axis=axis, batch_norm=batch_norm, activation=activation)(weights)
        return act

    return func


def _bn_act_conv_3D(**layer_params):
    """
        Block: BN -> Activation(Relu) -> Conv
        :param conv_params:
        :return:
    """

    channel_axis = _get_channel_axis()
    filters = layer_params['filters']
    kernel_size = layer_params['kernel_size']
    batch_norm = layer_params.setdefault('batch_norm', True)
    strides = layer_params.setdefault('strides', (1, 1, 1))
    dilation_rate = layer_params.setdefault('dilation_rate', 1)
    kernel_initializer = layer_params.setdefault('kernel_initializer', 'he_normal')
    padding = layer_params.setdefault('padding', 'same')
    kernel_regularizer = layer_params.setdefault('kernel_regularizer', l2(1.e-4))
    axis = layer_params.setdefault('axis', channel_axis)
    activation = layer_params.setdefault('activaton', 'relu')
    dropout = layer_params.setdefault('dropout', 0.0)
    train = layer_params.setdefault('train', False)

    def func(input_tensor):
        act = _bn_act(axis=axis, batch_norm=batch_norm, activation=activation)(input_tensor)
        weights = _conv_3D(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=padding,
                        dilation_rate=dilation_rate,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer)(act)

        if dropout > 0:
            weights = Dropout(rate=dropout)(weights, training=train)

        return weights

    return func


def _bn_act_deconv_3D(**layer_params):
    """
        Block: BN -> Activation(Relu) -> Conv
        :param conv_params:
        :return:
    """

    channel_axis = _get_channel_axis()
    filters = layer_params['filters']
    kernel_size = layer_params['kernel_size']
    strides = layer_params.setdefault('strides', (1, 1))
    dilation_rate = layer_params.setdefault('dilation_rate', 1)
    kernel_initializer = layer_params.setdefault('kernel_initializer', 'he_normal')
    padding = layer_params.setdefault('padding', 'same')
    kernel_regularizer = layer_params.setdefault('kernel_regularizer', l2(1.e-4))
    axis = layer_params.setdefault('axis', channel_axis)
    batch_norm = layer_params.setdefault('batch_norm', True)
    activation = layer_params.setdefault('activaton', 'relu')

    def func(input_tensor):
        act = _bn_act(axis=axis, batch_norm=batch_norm, activation=activation)(input_tensor)
        weights = _deconv_3D(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=padding,
                        dilation_rate=dilation_rate,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer)(act)
        return weights

    return func



def _make_bn_act_conv_block_3D(**layer_params):
    filters = layer_params['filters']
    kernel_size = layer_params['kernel_size']
    strides = layer_params.setdefault('strides', (1, 1, 1))
    dilation_rates = layer_params.setdefault('dilation_rates', [1, 1, 1])
    activation = layer_params.setdefault('activaton', 'relu')
    batch_norm = layer_params.setdefault('batch_norm', True)
    dropout = layer_params.setdefault('dropout', 0.0)
    train = layer_params.setdefault('train', False)

    def func(input_tensor):
        _, height, width, depth, _ = K.int_shape(input_tensor)

        # assert checking size power of 2
        assert height % 2 == 0
        assert width % 2 == 0
        assert depth % 2 == 0

        x = input_tensor
        x_list = []
        for ii in range(len(dilation_rates)):
            dilation_rate = dilation_rates[ii]
            x = _bn_act_conv_3D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
                                activation=activation, batch_norm=batch_norm, dropout=dropout, train=train)(x)
            x_list.append(x)

        x_pool = MaxPooling3D(pool_size=(2, 2, 2))(x)

        return x_pool, x, x_list

    return func


def _make_deconv_bn_act_block_3D(**layer_params):
    filters = layer_params['filters']
    kernel_size = layer_params['kernel_size']
    strides = layer_params.setdefault('strides', (1, 1, 1))
    dilation_rates = layer_params.setdefault('dilation_rates', [1, 1, 1])
    activation = layer_params.setdefault('activaton', 'relu')
    batch_norm = layer_params.setdefault('batch_norm', True)
    dropout = layer_params.setdefault('dropout', 0.0)

    def func(input_tensor, skip_tensor):

        # x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2))(input_tensor)
        x = _deconv_bn_act_3D(filters=filters, kernel_size=(2, 2, 2), strides=(2, 2, 2))(input_tensor)


        # # computation of cropping-amount needed for skip_tensor
        _, x_height, x_width, x_depth, _ = K.int_shape(x)
        _, s_height, s_width, s_depth, _ = K.int_shape(skip_tensor)

        h_crop = s_height - x_height
        w_crop = s_width - x_width
        d_crop = s_depth - x_depth

        assert h_crop >= 0
        assert w_crop >= 0
        assert d_crop >= 0
        if h_crop == 0 and w_crop == 0 and d_crop == 0:
            y = skip_tensor
        else:
            cropping = ((h_crop // 2, h_crop - h_crop // 2), (w_crop // 2, w_crop - w_crop // 2),
                        (d_crop // 2, d_crop - d_crop // 2))
            y = Cropping3D(cropping=cropping)(skip_tensor)

            # cropping = ((h_crop // 2, h_crop - h_crop // 2), (w_crop // 2, w_crop - w_crop // 2))
            # y = Cropping2D(cropping=cropping)(skip_tensor)


        # commented out at the moment because of error message:
        # could not create a view primitive descriptor, in file tensorflow/core/kernels/mkl_slice_op.cc:300

        x = Concatenate()([x, y])

        x_list = []
        for ii in range(len(dilation_rates)):
            dilation_rate = dilation_rates[ii]
            x = _conv_bn_act_3D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
                                activation=activation, batch_norm=batch_norm, dropout=dropout)(x)
            x_list.append(x)

        return x, x_list

    return func


def _make_conv_bn_act_block_3D(**layer_params):
    filters = layer_params['filters']
    kernel_size = layer_params['kernel_size']
    strides = layer_params.setdefault('strides', (1, 1, 1))
    dilation_rates = layer_params.setdefault('dilation_rates', [1, 1, 1])
    activation = layer_params.setdefault('activaton', 'relu')
    batch_norm = layer_params.setdefault('batch_norm', True)
    dropout = layer_params.setdefault('dropout', 0.0)
    train = layer_params.setdefault('train', False)

    def func(input_tensor):
        _, height, width, depth, _ = K.int_shape(input_tensor)
        assert height % 2 == 0
        assert width % 2 == 0
        assert depth % 2 == 0

        x = input_tensor
        x_list = []
        for ii in range(len(dilation_rates)):
            dilation_rate = dilation_rates[ii]
            x = _conv_bn_act_3D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
                                activation=activation, batch_norm=batch_norm, dropout=dropout, train=train)(x)
            x_list.append(x)

        x_pool = MaxPooling3D(pool_size=(2, 2, 2))(x)

        return x_pool, x, x_list

    return func


def _conv_bn_act_3D(**layer_params):
    """
        Block: BN -> Activation(Relu) -> Conv
        :param conv_params:
        :return:
    """

    channel_axis = _get_channel_axis()
    filters = layer_params['filters']
    kernel_size = layer_params['kernel_size']
    strides = layer_params.setdefault('strides', (1, 1, 1))
    dilation_rate = layer_params.setdefault('dilation_rate', 1)
    kernel_initializer = layer_params.setdefault('kernel_initializer', 'he_normal')
    padding = layer_params.setdefault('padding', 'same')
    kernel_regularizer = layer_params.setdefault('kernel_regularizer', l2(1.e-4))
    axis = layer_params.setdefault('axis', channel_axis)
    activation = layer_params.setdefault('activaton', 'relu')
    batch_norm = layer_params.setdefault('batch_norm', True)
    dropout = layer_params.setdefault('dropout', 0.0)
    train = layer_params.setdefault('train', False)

    def func(input_tensor):
        weights = _conv_3D(filters=filters, kernel_size=kernel_size,
                           strides=strides, padding=padding,
                           dilation_rate=dilation_rate,
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer)(input_tensor)
        act = _bn_act(axis=axis, batch_norm=batch_norm, activation=activation)(weights)

        if dropout > 0:
            act = Dropout(rate=dropout)(act, training=train)

        return act

    return func


def _make_bn_act_deconv_block_3D(**layer_params):
    filters = layer_params['filters']
    kernel_size = layer_params['kernel_size']
    strides = layer_params.setdefault('strides', (1, 1, 1))
    dilation_rates = layer_params.setdefault('dilation_rates', [1, 1, 1])
    activation = layer_params.setdefault('activaton', 'relu')
    batch_norm = layer_params.setdefault('batch_norm', True)
    dropout = layer_params.setdefault('dropout', 0.0)
    train = layer_params.setdefault('train', False)
    def func(input_tensor, skip_tensor):

        # x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2))(input_tensor)
        x = _bn_act_deconv_3D(filters=filters, kernel_size=(2, 2, 2), strides=(2, 2, 2))(input_tensor)

        # computation of cropping-amount needed for skip_tensor
        _, x_height, x_width, x_depth, _ = K.int_shape(x)
        _, s_height, s_width, s_depth, _ = K.int_shape(skip_tensor)

        h_crop = s_height - x_height
        w_crop = s_width - x_width
        d_crop = s_depth - x_depth

        assert h_crop >= 0
        assert w_crop >= 0
        assert d_crop >= 0
        if h_crop == 0 and w_crop == 0 and d_crop == 0:
            y = skip_tensor
        else:
            cropping = ((h_crop // 2, h_crop - h_crop // 2), (w_crop // 2, w_crop - w_crop // 2),
                        (d_crop // 2, d_crop - d_crop // 2))
            y = Cropping3D(cropping=cropping)(skip_tensor)

        x = Concatenate()([x, y])

        x_list = []
        for ii in range(len(dilation_rates)):
            dilation_rate = dilation_rates[ii]
            x = _bn_act_conv_3D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
                                activation=activation, batch_norm=batch_norm, dropout=dropout, train=train)(x)
            x_list.append(x)

        return x, x_list

    return func
