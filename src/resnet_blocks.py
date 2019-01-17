import keras

weight_decay = 1e-4

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    Modified to include an adapter module.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a',
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    # add adapter
    adapt = keras.layers.Conv2D(filters2, (1, 1),
                                kernel_initializer=keras.initializers.Orthogonal(gain=0.1),
                                kernel_regularizer=keras.regularizers.l2(weight_decay),
                                padding='same', name=conv_name_base + '2b_adapt')(x)
    x = keras.layers.Conv2D(filters2, kernel_size,
                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                            padding='same', name=conv_name_base + '2b')(x)
    x = keras.layers.add([x, adapt])

    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c',
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = keras.layers.add([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    Modified to include an adapter module.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    bn_axis = 3
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a',
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    # add adapter
    adapt = keras.layers.Conv2D(filters2, (1, 1), padding='same',
                                kernel_initializer=keras.initializers.Orthogonal(gain=0.1),
                                kernel_regularizer=keras.regularizers.l2(weight_decay),
                                name=conv_name_base + '2b_adapt')(x)
    x = keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b',
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.add([x, adapt])

    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c',
                            kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1',
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))(input_tensor)
    shortcut = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x
