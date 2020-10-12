from keras import *

from keras.objectives import *
from keras.callbacks import *
from keras.layers import Input, Activation, Reshape, Lambda
from keras.models import Model


# from layers import _conv, _mean_var_norm
# from layers import _make_bn_act_conv_block, _bn_act_conv, _make_deconv_bn_act_block
# from layers import _make_conv_bn_act_block, _conv_bn_act, _make_bn_act_deconv_block

from layers_3D import _conv_3D
from layers_3D import _make_bn_act_conv_block_3D, _bn_act_conv_3D, _make_deconv_bn_act_block_3D
from layers_3D import _make_conv_bn_act_block_3D, _conv_bn_act_3D, _make_bn_act_deconv_block_3D


from keras.layers import Dense

import keras.backend as K

from util import *

class UNet_3D:
    """Generates a U-Net model based on
          "U-Net: Convolutional Networks for Biomedical Image Segmentation"
          O. Ronneberger, P. Fischer, T. Brox (2015)
        Inputs:
          input_shape [height, width, channels]: input image height, width, number of channels: 1 for grayscale , 3 for RGB
          nb_labels - number of output classes
          depth  - number of downsampling operations in the encoder/decoder parts (4 in paper)
          nb_filters - number of output features for first convolution layer (64 in paper). after each down-sampling block, the no. of features get doubled.
          padding - 'valid' (used in paper) or 'same'
          batchnorm - include batch normalization layers before activations
          dropout_encoder - fraction of units  to dropout for the encoder_part, 0 to keep all units
          dropout_decoder - fraction of units to dropout for the decoder_part, 0 to keep all units
        Output:
          U-Net model expecting input shape (height, width, maps) and generates
          output with shape (output_height, output_width, classes). If padding is
          'same', then output_height = height and output_width = width.
        """
    def __init__(self, input_shape=[128, 128, 16, 1], nb_labels=2, depth=3, nb_bottleneck=3, filters=16, kernel_size=(3, 3, 3), dilation_rates=[1, 1],
                 padding='same', activation='relu', activation_network='softmax', batch_norm=True,
                 dropout_encoder=0.0, dropout_decoder=0.0, use_preact=False, use_mvn=False, train=False):
        self.input_shape = input_shape
        self.nb_labels = nb_labels
        self.depth = depth
        self.nb_bottleneck = nb_bottleneck
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.padding = padding
        self.activation = activation.lower()
        self.activation_network = activation_network.lower()
        self.batch_norm = batch_norm
        self.dropout_encoder = dropout_encoder
        self.dropout_decoder = dropout_decoder
        self.use_preact = use_preact
        self.use_mvn = use_mvn
        self.train = train

        #if K.image_dim_ordering() == 'th':
        #    self.axis = 1
        #else:
        self.axis = 3


    def create_model(self):

        inputs = Input(self.input_shape)
        im_rows, im_cols, im_slices, _ = self.input_shape
        nb_labels = self.nb_labels
        activation = self.activation
        kernel_size = self.kernel_size
        dilation_rates = self.dilation_rates
        padding = self.padding
        batch_norm = self.batch_norm
        use_preact = self.use_preact

        x = inputs
        #x = Lambda(lambda y: y / 255)(inputs)

        skip_layers = []

        filters = self.filters


        for dd in range(self.depth):

            if use_preact:
                x, x0, _ = _make_bn_act_conv_block_3D(filters=filters, kernel_size=kernel_size, dilation_rates=dilation_rates,
                                                   padding=padding, activation=activation, batch_norm=batch_norm,
                                                   dropout=self.dropout_encoder, train=self.train)(x)
            else:
                x, x0, _ = _make_conv_bn_act_block_3D(filters=filters, kernel_size=kernel_size, dilation_rates=dilation_rates,
                                                   padding=padding, activation=activation, batch_norm=batch_norm,
                                                   dropout=self.dropout_encoder, train=self.train)(x)
            skip_layers.append(x0)
            filters *= 2



        for bn in range(self.nb_bottleneck):
            if use_preact:
                x = _bn_act_conv_3D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation,
                                batch_norm=batch_norm, dropout=self.dropout_encoder)(x)
            else:
                x = _conv_bn_act_3D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation,
                                 batch_norm=batch_norm, dropout=self.dropout_encoder)(x)



        for dd in reversed(range(self.depth)):

            filters //= 2
            if use_preact:
                x, _ = _make_bn_act_deconv_block_3D(filters=filters, kernel_size=kernel_size, dilation_rates=dilation_rates,
                                                 padding=padding, activation=activation, batch_norm=batch_norm,
                                                 dropout=self.dropout_decoder, train=self.train)(x, skip_layers[dd])
            else:
                x, _ = _make_deconv_bn_act_block_3D(filters=filters, kernel_size=kernel_size, dilation_rates=dilation_rates,
                                                 padding=padding, activation=activation, batch_norm=batch_norm,
                                                 dropout=self.dropout_decoder, train=self.train)(x, skip_layers[dd])


        outputs = _conv_3D(filters=self.nb_labels, kernel_size=(1, 1, 1), activation='linear')(x)
        outputs = Reshape((im_rows * im_cols * im_slices, nb_labels))(outputs)
        outputs = Activation(self.activation_network)(outputs)
        outputs = Reshape((im_rows, im_cols, im_slices, nb_labels))(outputs)

        model = Model(inputs=inputs, outputs=outputs, name="3DUNET")
        model.summary()

        return model

#
# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1" #"2"
#     model = UNet(input_shape=(256, 256, 5)).create_model()
#     #plot_model(model, 'contextnet.png', show_shapes=True)
#
#     save_model(model, '.', 'unet')
