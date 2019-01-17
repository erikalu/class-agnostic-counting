from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras
import resnet_blocks
from l2_norm import L2_Normalization_Layer
from losses import center_weighted_mse_loss


def Synchronizing_model_weights(model1, model2):
    # This function is to assign the weights from model1 to model2.
    print('='*40)
    print('Start synchronizing the weights between two models.')
    print('='*40)
    for i, l in enumerate(model1.layers):
        model2.layers[i].set_weights(l.get_weights())
    return model2


def ResNet_share_architecture(inp, prefix=''):
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name=prefix + 'conv1')(inp)
    x = keras.layers.BatchNormalization(axis=-1, name=prefix + 'bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = resnet_blocks.conv_block(x, 3, [64, 64, 256], stage=2, block=prefix + 'a', strides=(1, 1))
    x = resnet_blocks.identity_block(x, 3, [64, 64, 256], stage=2, block=prefix + 'b')
    x = resnet_blocks.identity_block(x, 3, [64, 64, 256], stage=2, block=prefix + 'c')

    x = resnet_blocks.conv_block(x, 3, [128, 128, 512], stage=3, block=prefix + 'a')
    x = resnet_blocks.identity_block(x, 3, [128, 128, 512], stage=3, block=prefix + 'b')
    x = resnet_blocks.identity_block(x, 3, [128, 128, 512], stage=3, block=prefix + 'c')
    x = resnet_blocks.identity_block(x, 3, [128, 128, 512], stage=3, block=prefix + 'd')
    return x


def ResNet_base(input_dim):
    # ==> create model.
    img_input = keras.layers.Input(shape=input_dim, name='image')
    x = ResNet_share_architecture(inp=img_input)
    model = keras.models.Model(inputs=img_input, outputs=x, name='resnet_base')

    # ==> load pretrained resnet50 or not.
    pretrain_resnet50 = 'models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    if os.path.isfile(pretrain_resnet50):
        model.load_weights(os.path.join(pretrain_resnet50), by_name=True)
        print('==> successfully loaded pretrained resnet50')
    else:
        print('==> could not load pretrained resnet50')
    return model


def ResNet_patchnet(input_dim):
    prefix = 'patchnet_'
    # ==> create model.
    img_input = keras.layers.Input(shape=input_dim, name=prefix+'image')
    x = ResNet_share_architecture(inp=img_input, prefix=prefix)
    model = keras.models.Model(inputs=img_input, outputs=x, name='resnet50_patchnet')
    return model


def matching_net(inputs, prefix='upsample'):
    outputs = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name=prefix + '_conv_1')(inputs)
    outputs = keras.layers.BatchNormalization(name=prefix + '_bn_1')(outputs)
    outputs = keras.layers.Activation('relu')(outputs)
    # ==> upsample
    outputs = keras.layers.Conv2DTranspose(256, (3, 3), strides=(2,2), padding='same', name=prefix+'_convt_1')(outputs)
    outputs = keras.layers.BatchNormalization(name=prefix + '_bn_2')(outputs)
    outputs = keras.layers.Activation('relu')(outputs)
    return outputs


#-----------------------------
# ==>   MAIN NETWORK
#-----------------------------
def two_stream_matching_networks(cg, adapt=False, sync=True):
    dim1 = cg.patchdims
    dim2 = cg.imgdims
    inputs = []
    inputs += [keras.layers.Input(shape=dim1, name='image_patch')]
    inputs += [keras.layers.Input(shape=dim2, name='image')]

    patchnet = ResNet_patchnet(input_dim=dim1)
    basenet = ResNet_base(input_dim=dim2)

    if sync:
        patchnet = Synchronizing_model_weights(model1=basenet, model2=patchnet)

    # ==> pass the exemplar patch to patchnet,
    # the patchnet will output a small feature map (M' x N' x channels),
    # take a global avgpooling, making the feature maps (1 x 1 x channels)
    exemplar = patchnet(inputs[0])
    exemplar = keras.layers.GlobalAveragePooling2D()(exemplar)

    # ==> pass the image to a resnet-style networks,
    # ==> output a feature map with (W' x H' x channels)
    image_f = basenet(inputs[1])

    # ==> L2 normalize
    exemplar = L2_Normalization_Layer(name='exemplar_l2')(exemplar)
    image_f = L2_Normalization_Layer(name='image_f_l2')(image_f)

    # ==> broadcast exemplar feature vector (1 x 1 x 512) to same size as image features
    exemplar = keras.layers.RepeatVector(image_f.shape[1].value*image_f.shape[2].value)(exemplar)
    exemplar = keras.layers.Reshape(image_f.get_shape().as_list()[1:4])(exemplar)

    # ==> oncatenate exemplar and image features
    outputs = keras.layers.Concatenate(axis=-1)([exemplar, image_f])

    # ==> matching module
    outputs = matching_net(outputs)

    # ==> layer to produce final heatmap
    outputs = keras.layers.Conv2D(1, (3, 3), strides=(1, 1), activation='relu', padding='same', name='output')(outputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    # ==> adapters
    layers = [l.layers if isinstance(l, model.__class__) else [l] for l in model.layers]
    layers = [l for ls in layers for l in ls]
    if adapt:
        # freeze all layers except batchnorm, adapter, and l2 normalization scaling
        # bn is trick, because of the disparity of training and inference mode
        for layer in layers:
            if not (isinstance(layer, type(keras.layers.BatchNormalization()))
                    or 'adapt' in layer.name 
                    or 'l2' in layer.name):
                layer.trainable = False
            #else:
            #    print('train: %s' % layer.name)
    else:
        # freeze adapters
        for layer in layers:
            if 'adapt' in layer.name:
                layer.trainable = False
                #print('freeze: %s' % layer.name)

    # ==> compile model
    opt = keras.optimizers.Adam(1e-3)
    model.compile(optimizer=opt, loss=center_weighted_mse_loss)
    return model

