import os
import random
import numpy as np
import scipy.ndimage
import skimage.measure


def initialize_GPU(args):
    # Initialize GPUs
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def set_path(args):
    if args.mode == 'pretrain':
        import datetime
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        exp_path = os.path.join(args.mode, '{0}_{args.mode}_{args.net}_{args.dataset}'
                                           '_{args.optimizer}_lr{args.lr}_bs{args.batch_size}'.format(date, args=args))
    else:
        exp_path = os.path.join(args.mode, args.gmn_path.split(os.sep)[-2])
    model_path = os.path.join('models', exp_path)
    log_path = os.path.join('logs', exp_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    return model_path, log_path


def step_decay(args):
    def step_decay_fn(epoch):
        '''
        The learning rate begins at 10^initial_power,
        and decreases by a factor of 10 every step epochs.
        '''
        stage1, stage2, stage3 = int(args.epochs * 0.4), int(args.epochs * 0.7), args.epochs
        if args.warmup_ratio:
            milestone = [2, stage1, stage2, stage3]
            gamma = [args.warmup_ratio, 1.0, 0.1, 0.01]
        else:
            milestone = [stage1, stage2, stage3]
            gamma = [1.0, 0.1, 0.01]
        lr = 0.0005
        init_lr = args.lr
        stage = len(milestone)
        for s in range(stage):
            if epoch < milestone[s]:
                lr = init_lr * gamma[s]
                break
        print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
        return np.float(lr)
    return step_decay_fn


def max_pooling(img, stride=(2, 2)):
    return skimage.measure.block_reduce(img, block_size=stride, func=np.max)


def flip_axis(array, axis):
    """
    Flip the given axis of an array.  Note that the ordering follows the
    numpy convention and may be unintuitive; that is, the first axis
    flips the axis horizontally, and the second axis flips the axis vertically.
    :param array: The array to be flipped.
    :type array: `ndarray`
    :param axis: The axis to be flipped.
    :type axis: `int`
    :returns: The flipped array.
    :rtype: `ndarray`
    """

    # Rearrange the array so that the axis of interest is first.
    array = np.asarray(array).swapaxes(axis, 0)
    # Reverse the elements along the first axis.
    array = array[::-1, ...]
    # Put the array back and return.
    return array.swapaxes(0, axis)


def affine_transform_Image(img, matrix, offset):
    #padX = [img.shape[1] - pivot[0], pivot[0]]
    #padY = [img.shape[0] - pivot[1], pivot[1]]
    #imgP = np.pad(img, [padY, padX, [0,0]], 'reflect')
    imgR = scipy.ndimage.affine_transform(img, matrix, offset=offset, mode='nearest', order=5)
    return imgR


def affine_image_with_python(img, target_shape=None, xy=np.array([0.0, 0.0]), rt=0.0, zm=1.0):
    # This is specifically designed for the stn face project.
    xy_mat = np.array([1.0, 1.0, 1.0, 1.0])
    rt_mat = np.array([np.cos(rt), np.sin(rt), -np.sin(rt), np.cos(rt)])
    zm_mat = np.array([zm, zm, zm, zm])
    transform_mat = np.reshape((xy_mat * rt_mat) * zm_mat, (2, 2))
    c_in = 0.5*np.array(img.shape[:2])
    c_out = c_in
    offset = c_in - c_out.dot(transform_mat)
    trans_img_c0 = affine_transform_Image(img[:, :, 0], transform_mat.T, offset=offset+xy*(target_shape[:2]//2))
    trans_img_c1 = affine_transform_Image(img[:, :, 1], transform_mat.T, offset=offset+xy*(target_shape[:2]//2))
    trans_img_c2 = affine_transform_Image(img[:, :, 2], transform_mat.T, offset=offset+xy*(target_shape[:2]//2))
    trans_img = np.stack((trans_img_c0, trans_img_c1, trans_img_c2), -1)
    return trans_img


def load_data(imgpath, dims=None, pad=0, normalize=False):
    '''
    dims: desired output shape
    pad (int): pixels of mean padding to include on each border
    normalize: if True, return image in range [0,1]
    '''
    img = scipy.misc.imread(imgpath, mode='RGB')
    if normalize:
        img = img/255.
    if dims:
        imgdims = (dims[0]-pad*2, dims[1]-pad*2, dims[2])
        img = scipy.misc.imresize(img, (imgdims[0], imgdims[1]))
        if pad:
            padded_im = np.zeros(dims)
            padded_im[:] = np.mean(img, axis=(0, 1))
            padded_im[pad:imgdims[0]-pad, pad:imgdims[1]-pad, :] = img

    return img


def load_dotlabel(lbpath, imgdims, pad=0):
    '''
    load labels stored as dot annotation maps
    imgdims: output size
    pad (int): pixels of zero padding to include on each border
    '''

    lb = scipy.misc.imread(lbpath, mode='RGB')

    # resize dot labels
    lb = np.asarray(lb[:, :, 0] > 230)
    coords = np.column_stack(np.where(lb == 1))
    new_lb = np.zeros((imgdims[0], imgdims[1]), dtype='float32')

    zx = (imgdims[0]-2*pad)/lb.shape[0]
    zy = (imgdims[1]-2*pad)/lb.shape[1]

    for c in range(coords.shape[0]):
        new_lb[pad+int(coords[c,0]*zx),pad+int(coords[c, 1]*zy)] = 1

    return new_lb


def sample_exemplar(inputs, patchdims, augment):
    '''
    Samples an exemplar patch from an input image.
    Args:
        inputs: tuple of (img, lb)
            img: input image
            lb: dot annotations of instances (same size as img)
        patchdims: desired size of exemplar patch
        augment: whether to do data augmentation on patch
    '''
    img,lb = inputs
    imgdims = img.shape

    # get coordinates of potential exemplars
    coords = np.column_stack(np.where(lb == 1.0))
    valid_coords = np.array([c for c in coords
                             if (c[0] > patchdims[0]//2) and c[1] > (patchdims[1]//2)
                             and c[0] < (imgdims[0] - patchdims[0]//2)
                             and c[1] < (imgdims[1] - patchdims[1]//2)])

    if valid_coords.shape[0] == 0:
        # TODO: different way of handling this case
        # no objects, so choose patch at center of image to match to itself
        valid_coords = np.array([[imgdims[0] // 2, imgdims[1] // 2]], 'int')
        lb[:] = 0
        lb[valid_coords[0][0], valid_coords[0][1]] = 1

    patch_coords = valid_coords[random.randint(0, valid_coords.shape[0]-1)]
    ex_patch = img[patch_coords[0] - patchdims[0] // 2: patch_coords[0] + patchdims[0] // 2,
                   patch_coords[1] - patchdims[1] // 2: patch_coords[1] + patchdims[1] // 2, ]

    output_map = max_pooling(lb, (4, 4))  # resize to output size
    output_map = 100 * scipy.ndimage.gaussian_filter(
            output_map, sigma=(2, 2), mode='constant')

    if augment:
        opt = {'xy': -0.05, 'rt': [1, 20], 'zm': [0.9, 1.1]}
        ex_patch = augment_data(ex_patch, opt)
    return (ex_patch, output_map)


def augment_data(img, opt={}, prob=.9):
    '''
    performs a random horizontal flip
    and a random affine transform with probability prob
    Args:
        opt: options for adjusting amount of translation, rotation, zoom
    '''

    xy = opt.get('xy', -0.03)
    rt = opt.get('rt', [8, 20])
    zm = opt.get('zm', [.95, 1.05])

    if random.random() > .5:
        img = flip_axis(img, 1)

    if random.random() < prob:
        rand_xy = xy * np.random.random((2,))
        rand_rt = np.pi / random.randint(rt[0], rt[1])
        rand_zm = np.random.uniform(zm[0], zm[1])
        target_shape = np.array(img.shape)

        img = affine_image_with_python(img, target_shape, xy=rand_xy, rt=rand_rt, zm=rand_zm)
    return img


def multiprocess_fn(pool, fn, input_list, opts=[]):
    results = [pool.apply_async(fn, args=(x,)+tuple(opts)) for x in input_list]
    results = [p.get() for p in results]
    return results
