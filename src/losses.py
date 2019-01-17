from keras import backend as K
import numpy as np


def center_weighted_mse_loss(y_true, y_pred):
    mask = K.random_normal(K.shape(y_true))
    mask = K.cast(K.greater(mask, 0.8), 'float32')
    # weigh the center more
    radius=8
    center = np.zeros((K.int_shape(y_pred)[1],K.int_shape(y_pred)[2],1))
    center[K.int_shape(y_pred)[1] // 2 - radius : K.int_shape(y_pred)[1] // 2 + radius,
          K.int_shape(y_pred)[2] // 2 - radius : K.int_shape(y_pred)[2]// 2 + radius,:] = 1
    center =  mask + 10*K.variable(center)
    return K.mean(center * K.square(y_pred - y_true), axis=-1)


