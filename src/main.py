import os
import sys
import argparse
import numpy as np
import utils as ut

# ===========================================
#        Parse the argument
# ===========================================
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--net', default='resnet50', choices=['resnet50'], type=str)
parser.add_argument('--optimizer', default='adam', choices=['adam'], type=str)
parser.add_argument('--mode', default='pretrain', choices=['pretrain', 'adapt'], type=str,
                    help='pretrain on tracking data or adapt to specific dataset.')
parser.add_argument('--dataset', default='imagenet',
                    choices=['imagenet', 'vgg_cell', 'hela_cell', 'car'],
                    type=str, help='pretrain on tracking data or adapt to specific dataset.')
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--warmup_ratio', default=0, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--gmn_path', default='', type=str,
                    help='path to pretrained GMN, used for "adapt" mode')
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--data_path', default='', type=str)
parser.add_argument('--epochs', default=36, type=int,
                    help='number of total epochs to run')

global args
args = parser.parse_args()

def train_gmn():

    # ==> initial check
    assert args.dataset == 'imagenet'

    # ==> gpu configuration
    ut.initialize_GPU(args)

    # ==> set up model path and log path.
    model_path, log_path = ut.set_path(args)

    # ==> import library
    import keras
    import data_loader
    import model_factory
    import data_generator

    # ==> get dataset information
    trn_config = data_loader.get_config(args)
    params = {'cg': trn_config,
              'processes': 12,
              'batch_size': args.batch_size,
              }
    trn_gen, val_gen = data_generator.setup_generator(**params)

    # ==> load model
    gmn = model_factory.two_stream_matching_networks(trn_config)
    gmn.summary()

    # ==> attempt to load pre-trained model
    if args.resume:
        if os.path.isfile(args.resume):
            gmn.load_weights(os.path.join(args.resume), by_name=True)
            print('==> successfully loading the model: {}'.format(args.resume))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    # ==> set up callbacks, e.g. lr schedule, tensorboard, save checkpoint.
    normal_lr = keras.callbacks.LearningRateScheduler(ut.step_decay(args))
    tbcallbacks = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=False, write_images=False)
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'model.h5'),
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 mode='min'),
                 normal_lr, tbcallbacks]

    gmn.fit_generator(trn_gen,
                      steps_per_epoch=600,
                      epochs=args.epochs,
                      validation_data=val_gen,
                      validation_steps=100,
                      callbacks=callbacks,
                      verbose=1)


def adapt_gmn():

    # ==> gpu configuration
    ut.initialize_GPU(args)

    # ==> set up model path and log path.
    model_path, log_path = ut.set_path(args)

    # ==> import library
    import keras
    import data_loader
    import model_factory
    import data_generator

    # ==> get dataset information
    trn_config = data_loader.get_config(args)

    params = {'cg': trn_config,
              'processes': 12,
              'batch_size': args.batch_size
              }

    trn_gen, val_gen = data_generator.setup_generator(**params)

    # ==> load networks
    gmn = model_factory.two_stream_matching_networks(trn_config, sync=False, adapt=False)
    model = model_factory.two_stream_matching_networks(trn_config, sync=False, adapt=True)

    # ==> attempt to load pre-trained model
    if args.resume:
        if os.path.isfile(args.resume):
            model.load_weights(os.path.join(args.resume), by_name=True)
            print('==> successfully loading the model: {}'.format(args.resume))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    # ==> attempt to load pre-trained GMN
    elif args.gmn_path:
        if os.path.isfile(args.gmn_path):
            gmn.load_weights(os.path.join(args.gmn_path), by_name=True)
            print('==> successfully loading the model: {}'.format(args.gmn_path))
        else:
            print("==> no checkpoint found at '{}'".format(args.gmn_path))

    # ==> print model summary
    model.summary()

    # ==> transfer weights from gmn to new model (this step is slow, but can't seem to avoid it)
    for i,layer in enumerate(gmn.layers):
        if isinstance(layer, model.__class__):
            for l in layer.layers:
                weights = l.get_weights()
                if len(weights) > 0:
                    #print('{}'.format(l.name))
                    model.layers[i].get_layer(l.name).set_weights(weights)
        else:
            weights = layer.get_weights()
            if len(weights) > 0:
                #print('{}'.format(layer.name))
                model.get_layer(layer.name).set_weights(weights)

    # ==> set up callbacks, e.g. lr schedule, tensorboard, save checkpoint.
    normal_lr = keras.callbacks.LearningRateScheduler(ut.step_decay(args))
    tbcallbacks = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=False, write_images=False)
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(model_path, 'model.h5'),
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 mode='min'),
                 normal_lr, tbcallbacks]

    model.fit_generator(trn_gen,
                        steps_per_epoch=600,
                        epochs=args.epochs,
                        validation_data=val_gen,
                        validation_steps=100,
                        callbacks=callbacks,
                        verbose=1)


if __name__ == '__main__':
    if args.mode == 'pretrain':
        train_gmn()
    elif args.mode == 'adapt':
        adapt_gmn()
    else:
        raise IOError('==> will not happen.')
