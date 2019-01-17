# Class-Agnostic Counting

This repo contains a Keras implementation of the paper,     [Class-Agnostic Counting (Lu et al., ACCV 2018)](https://arxiv.org/abs/1811.00472). It includes code for training the GMN (Generic Matching Network) and adapting it to specific datasets.
 
### Dependencies
- [Python 3.4.9](https://www.python.org/downloads/)
- [Keras 2.1.5](https://keras.io/)
- [Tensorflow 1.6.0](https://www.tensorflow.org/)


### Data
Download and preprocess the data for training the GMN following the instructions at: https://github.com/bertinetto/siamese-fc/tree/master/ILSVRC15-curation [1]. 
Before preprocessing the dataset, change the following variables:
```
    exemplar_size = 63;
    instance_size = 255;
    context_amount = 0.1;
```
The following datasets were used for the adaptation experiments:
- [VGG synthetic cells](http://www.robots.ox.ac.uk/~vgg/research/counting/index_org.html) [2]
- [HeLa cells](http://www.robots.ox.ac.uk/~vgg/software/cell_detection/) [3]
- [CARPK cars](https://lafi.github.io/LPN/) [4]

Labels should be in the form of dot annotation images.

### Training the GMN
To train the Generic Matching Network (GMN) on the ImageNet video data, run

`python src/main.py --mode pretrain --data_path /path/to/ILSVRC2015_crops/train/`

The code expects ImageNet pretrained Resnet50 weights at

`models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5`

### Adapting the GMN
To adapt a trained GMN to a specific dataset, e.g. vgg cells, run

`python src/main.py --mode adapt --dataset vgg_cell --data_path /path/to/data --gmn_path /path/to/pretrained_gmn_model`

### References
```
[1] L. Bertinetto, J. Valmadre, J.F. Henriques, A. Vedaldi, P.H.S. Torr. "Fully-Convolutional Siamese Networks for Object Tracking." In ECCV Workshop 2016.
[2] V. Lempitsky and A. Zisserman. "Learning to Count Objects in Images." In NIPS 2010.
[3] C. Arteta, V. Lempitsky, J. A. Noble, A. Zisserman. "Learning to Detect Cells Using Non-overlapping Extremal Regions." In MICCAI 2012.
[4] M. Hsieh, Y. Lin, W. Hsu. "Drone-based Object Counting by Spatially Regularized Regional Proposal Networks." In ICCV 2017.
[5] W. Xie, J. A. Noble, A. Zisserman. "Microscopy Cell Counting with Fully Convolutional Regression Networks." In MICCAI Workshop 2016.
```

### Citation
```
@InProceedings{Lu18,
  author       = "Lu, E. and Xie, W. and Zisserman, A.",
  title        = "Class-agnostic Counting",
  booktitle    = "Asian Conference on Computer Vision",
  year         = "2018",
}
```

