import numpy as np

class get_config(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.datapath = args.data_path
        # data list
        self.trn_lst = None
        self.trn_lb = None
        self.val_lst = None
        self.val_lb = None
        # data dimension
        self.imgdims = None
        self.patchdims = None
        self.outputdims = None
        # amount to pad input images,
        # useful for resolving undesired boundary effects
        self.pad = None
        self.__set__()

    def __set__(self):
        if self.dataset == 'imagenet':
            # data preprocessing follows:
            # https://github.com/bertinetto/siamese-fc/tree/master/ILSVRC15-curation
            # reference:
            # L. Bertinetto, J. Valmadre, J.F. Henriques, A. Vedaldi, P.H.S. Torr,
            # "Fully-Convolutional Siamese Networks for Object Tracking", In ECCV16 Workshop.
            # the object is always centered
            self.imgdims = (255, 255, 3)
            self.patchdims = (63, 63, 3)
            self.outputdims = (64, 64, 1)
            self.patch_start = 127-32
            self.patch_end = 127+31
            self.pad = 0

        elif self.dataset == 'vgg_cell':
            self.imgdims = (800, 800, 3)
            self.patchdims = (64, 64, 3)
            self.outputdims = (200, 200, 1)
            self.pad = 0

        elif self.dataset == 'hela_cell':
            self.imgdims = (800, 800, 3)
            self.patchdims = (64, 64, 3)
            self.outputdims = (200, 200, 1)
            self.pad = 0

        elif self.dataset == 'car':
            self.imgdims = (360, 640, 3)
            self.patchdims = (64, 64, 3)
            self.outputdims = (90, 160, 1)
            self.pad = 0

        elif self.dataset == 'crowd':
            self.imgdims = (256, 256, 3)
            self.patchdims = (128, 128, 3)
            self.outputdims = (64, 64, 1)
            self.pad = 0

        else:
            raise IOError('==> unknown data set.')

        with np.load('meta/{}.npz'.format(self.dataset)) as data:
            self.trn_lst = data['trn_lst']
            self.trn_lb = data['trn_lb']
            self.val_lst = data['val_lst']
            self.val_lb = data['val_lb']

