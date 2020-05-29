# What have I done, and other stories

This directory was main large part forked from tensoflows [models](https://github.com/tensorflow/models/tree/master/research/deeplab) directory. Specifically just the slim and deeplab directories, which have since been marginally tweaked.
The whole project was not forked due to the large number of models in the directory.


## Work so far

This repo contained the large majority of the work conducted thus far attempt to train/tweak the deeplab model, to better recognise and then approriate segment areas of waterlogged grassland in fields. 

The work closely follows [heaversm tutorial](https://github.com/heaversm/deeplab-training), which is one of the best and most up to date tutorials in this currently. 

### Notes on the dataset
In order to train a dataset, 200 inidivual images were segmented (using labelbox) and then flipped, mirrored and flipped again to result in 800 images and segmentation ground truth masks. These would usually reside in the datasets/PQR/JPEGImages and datasets/PQR/SegmentationClassRaw directories, but have been ignored in this repo for size reasons.

Similarly, the evaluation and training checkpoint files have been omitted for issues of size.

The trained model, created from the dataset and TRFrecords, can be found under deeplab/datasets/trained_model.tar.gz