import sys
sys.path.append('./src')
import numpy as np
from collections import namedtuple
from PIL import Image

from data_generator import preprocess_input
from model_factory import two_stream_matching_networks


def inference(args):
    # Load image and exemplar patch.
    im = Image.open(args.im).convert('RGB')
    vis_im = im.resize((im.size[0]//4, im.size[1]//4))
    im = np.array(im)
    vis_im = np.array(vis_im)
    patch = np.array(Image.open(args.exemplar).convert('RGB'))
    if patch.shape[0] != 63 or patch.shape[1] != 63:
        raise Exception('The exemplar patch should be size 63x63.')

    # set up data
    im_pre = preprocess_input(im[np.newaxis, ...].astype('float'))
    patch_pre = preprocess_input(patch[np.newaxis, ...].astype('float'))
    data = {'image': im_pre,
            'image_patch': patch_pre }
    vis_im = vis_im / 255.

    # load trained model
    Config = namedtuple('Config', 'patchdims imgdims outputdims')
    cg = Config(patchdims=(63, 63, 3), imgdims=im.shape, outputdims=(im.shape[0]//4, im.shape[1]//4, 3))
    model = two_stream_matching_networks(cg, sync=False, adapt=False)
    model.load_weights('./checkpoints/pretrained_gmn.h5')
    # model.summary()

    # inference
    pred = model.predict(data)[0, :vis_im.shape[0], :vis_im.shape[1]]
    print('Count by summation: %0.2f' % (pred.sum()/100.))

    vis_im *= .5
    vis_im[..., 1] += pred[..., 0]/5.
    vis_im = np.clip(vis_im, 0, 1)
    vis_im = (vis_im*255).astype(np.uint8)
    vis_im = Image.fromarray(vis_im)
    outpath = 'heatmap_vis.jpg'
    vis_im.save(outpath)
    print('Predicted heatmap visualization saved to %s' % outpath)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--im', default='images/cells.jpg', type=str, help='path to image')
    parser.add_argument('--exemplar', default='images/exemplar_cell.jpg', type=str, help='path to exemplar patch')
    args = parser.parse_args()

    inference(args)
