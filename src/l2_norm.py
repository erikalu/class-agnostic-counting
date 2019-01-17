from __future__ import absolute_import
from __future__ import division
import keras

# ===================================================
#                Normalization Layer
# ===================================================
class L2_Normalization_Layer(keras.engine.Layer):
    '''
        This layer does L2 Normalization.
    '''
    def __init__(self, **kwargs):
        self.scale = True
        super(L2_Normalization_Layer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        if self.scale:
            self.alpha = self.add_weight(shape=[1, ],
                                         name='alpha',
                                         initializer=keras.initializers.RandomUniform(minval=10.0, maxval=20.0, seed=None))
        else:
            self.alpha = 1.0

    def call(self, x):
        return self.alpha * keras.backend.l2_normalize(x, -1)

