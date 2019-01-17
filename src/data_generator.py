# System
import os
import signal
import random
import multiprocessing as mp

# Third Party
import numpy as np
import scipy.ndimage
import keras.backend as K

import utils as ut


def setup_generator(processes=None, batch_size=10, cg=None):

    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except:
        pass

    trn_list, trn_lb, val_list, val_lb = cg.trn_lst, cg.trn_lb, cg.val_lst, cg.val_lb

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None

    if cg.dataset == 'imagenet':
        trn_gen = datagen_gmn(X=trn_list,
                              y=trn_lb,
                              cg=cg,
                              augment=True,
                              batch_size=batch_size, pool=pool)
    else:
        trn_gen = datagen_adapt(X=trn_list,
                                y=trn_lb,
                                cg=cg,
                                augment=True,
                                batch_size=batch_size, pool=pool)
    if val_list is None:
        val_gen = None
    else:
        if cg.dataset == 'imagenet':
            val_gen = datagen_gmn(X=val_list,
                                  y=val_lb,
                                  cg=cg,
                                  augment=False,
                                  batch_size=batch_size, pool=pool)
        else:
            val_gen = datagen_adapt(X=val_list,
                                    y=val_lb,
                                    cg=cg,
                                    augment=False,
                                    batch_size=batch_size, pool=pool)
    return trn_gen, val_gen


def datagen_gmn(X, y, cg, augment=True, batch_size=32, pool=None):
    '''
    Data has been preprocessed following the precedure:
    https://github.com/bertinetto/siamese-fc/tree/master/ILSVRC15-curation

    Process ImageNet video data for training the GMN.

    Args:
        X: numpy array with 1 entry per class (30 total for Imagenet video)
           each entry is a list of tuples that represent a single object in a video
           each tuple looks like: ( video_dir_path, object_id, [valid frame numbers] )
    '''

    listlength = X.shape[0]
    inp_img = np.zeros((batch_size, ) + cg.imgdims)
    ex_patch = np.zeros((batch_size, ) + cg.patchdims)
    output_map = np.zeros((batch_size, ) + cg.outputdims)

    # generate positive sample heatmap
    positive = np.zeros(cg.outputdims)
    positive[cg.outputdims[0] // 2, cg.outputdims[1] // 2, 0] = 1
    positive[:, :, 0] = 100 * scipy.ndimage.gaussian_filter(
                        positive[:, :, 0], sigma=(2, 2), mode='constant')

    while True:
        # select random classes
        sample = np.random.choice(listlength, batch_size)

        inp_img_paths = np.empty(batch_size, dtype='S120')
        ex_patch_paths = np.empty(batch_size, dtype='S120')
        output_map = np.zeros((batch_size, ) + cg.outputdims)

        for k, class_i in enumerate(sample):
            # first select a random (vid,obj) pair from each class
            obj = np.random.choice(X[class_i])

            if random.random() < 0.5:
                # positive pair
                # input image and patch are same object
                inp_obj = obj
                patch_obj = obj

                # choose 2 frames at most 100 frames apart
                start = np.random.randint(max(1, len(obj[2])-100))
                inp_frame, ex_frame = np.random.choice(obj[2][start:start+100], 2)

                # update output heatmap
                output_map[k] = positive
            else:
                # sample negative pair
                # input image and patch are different objects
                inp_obj = obj

                # choose random other class
                class_other = np.random.choice(
                        np.concatenate(
                            (np.arange(class_i),
                                np.arange(class_i+1, listlength))
                            ))
                # random object from other class
                patch_obj = np.random.choice(X[class_other])

                # random frame for each object 
                inp_frame = np.random.choice(inp_obj[2])
                ex_frame = np.random.choice(patch_obj[2])

            inp_img_paths[k] = os.path.join(cg.datapath,
                        '%s/%06d.%02d.crop.x.jpg' % (
                            inp_obj[0], inp_frame, inp_obj[1]))

            if augment:
                # need to read full image for later augmentation
                ex_patch_paths[k] = os.path.join(cg.datapath,
                        '%s/%06d.%02d.crop.x.jpg' % (
                            patch_obj[0], ex_frame, patch_obj[1]))
            else:
                # read patch directly
                ex_patch_paths[k] = os.path.join(cg.datapath,
                        '%s/%06d.%02d.crop.z.jpg' % (
                            patch_obj[0], ex_frame, patch_obj[1]))

        inp_img = ut.multiprocess_fn(pool, ut.load_data, inp_img_paths, [cg.imgdims,cg.pad,])

        if augment:
            # read full image, augment, crop patch
            ex_patch = ut.multiprocess_fn(pool, ut.load_data, ex_patch_paths, [cg.imgdims,cg.pad])
            ex_patch = ut.multiprocess_fn(pool, ut.augment_data, ex_patch)
            ex_patch = np.array(ex_patch)
            # crop center patch from full image
            ex_patch = ex_patch[:, cg.patch_start:cg.patch_end, cg.patch_start:cg.patch_end, :]
        else: 
            ex_patch = ut.multiprocess_fn(pool, ut.load_data, ex_patch_paths, [cg.patchdims])

        inp_img = preprocess_input(np.array(inp_img, dtype='float32'))
        ex_patch = preprocess_input(np.array(ex_patch, dtype='float32'))

        inputs = {'image_patch': ex_patch,
                  'image': inp_img,
                  }
        outputs = {'output': output_map}
        yield (inputs, outputs)


def datagen_adapt(X, y, cg, augment=True, batch_size=25, pool=None, cross_image=False):
    listlength = X.shape[0]

    inp_img = np.zeros((batch_size, ) + cg.imgdims)
    ex_patch = np.zeros((batch_size, ) + cg.patchdims)
    output_map = np.zeros((batch_size, ) + cg.outputdims)

    while True:
        sample = np.random.choice(listlength, batch_size)
        imglist = np.array([os.path.join(cg.datapath, X[t]) for t in sample])
        lblist = np.array([os.path.join(cg.datapath, y[t]) for t in sample])

        imgs = ut.multiprocess_fn(pool, ut.load_data, imglist, [cg.imgdims,cg.pad])
        lbs = ut.multiprocess_fn(pool, ut.load_dotlabel, lblist, [cg.imgdims,cg.pad])

        # Sample exemplar patch from full image and augment.
        results = ut.multiprocess_fn(pool, ut.sample_exemplar, list(zip(imgs, lbs)), [cg.patchdims, augment])
        ex_patch, output_map = list(zip(*results))

        if cross_image:
            if random.random() < 0.9:
                # shuffle patches so that they can be sampled from different images
                ex_patch = list(ex_patch)
                random.shuffle(ex_patch)

        inp_img = preprocess_input(np.array(imgs, dtype='float32'))
        ex_patch = preprocess_input(np.array(ex_patch, dtype='float32'))
        output_map = np.array(output_map)
        output_map = np.expand_dims(output_map, -1)

        inputs = {'image_patch': ex_patch,
                  'image': inp_img,
                  }
        outputs = {'output': output_map}
        yield (inputs, outputs)


def preprocess_input(x, dim_ordering='default'):
    '''
    imagenet preprocessing
    '''
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


