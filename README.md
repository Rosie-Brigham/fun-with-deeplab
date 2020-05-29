# Fun with Deeplab
This repo contained the large majority of the work conducted thus far attempt to train/tweak the deeplab model, to better recognise and then approriate segment areas of waterlogged grassland in fields. 

The work closely follows [heaversm tutorial](https://github.com/heaversm/deeplab-training), which is one of the best and most up to date tutorials I have found thus far.

This directory was main large part forked from tensoflows [models](https://github.com/tensorflow/models/tree/master/research/deeplab) directory. Specifically just the slim and deeplab directories, which have since been marginally tweaked.
The whole project was not forked due to the large number of models in the directory.


### Notes on the dataset
In order to train a dataset, 200 inidivual images were segmented (using labelbox) and then flipped, mirrored and flipped again to result in 800 images and segmentation ground truth masks. These would usually reside in the datasets/PQR/JPEGImages and datasets/PQR/SegmentationClassRaw directories, but have been ignored in this repo for size reasons.

Similarly, the evaluation and training checkpoint files have been omitted for issues of size.

The trained model, created from the dataset and TRFrecords, can be found under deeplab/datasets/trained_model.tar.gz


**Generate the tfrecord folder**

Tensorflow has a `tfrecord` format that makes storing training data much more efficient. This needs to be generated from the dataset (ignored on github, you will have to add this separately). To do so, this repo has made a copy of the `build_voc2012_data.py` file which has been saved as `build_pqr_data.py`. 

This has been edited to ensure to flag the manually segmented images. In this case, look at ~line80:

```python
tf.app.flags.DEFINE_string('image_folder',
                     './PQR/JPEGImages',
                     'Folder containing images.')

tf.app.flags.DEFINE_string(
'semantic_segmentation_folder',
'./PQR/SegmentationClassRaw',
'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
'list_folder',
'./PQR/ImageSets',
'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
'output_dir',
'./PQR/tfrecord',
'Path to save converted SSTable of TensorFlow examples.')
```

In this instance, the `tfrecord` folder **should** exist. The script will not make it for you. Also note that at around Line 119 I have hardcoded the input format to be `.jpg:

```python
image_filename = os.path.join(
#MH:
#FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
FLAGS.image_folder, filenames[i] + '.jpg')
#END MH
```

and the output images to be `.png`

```python
#MH:
      #filenames[i] + '.' + FLAGS.label_format)
      filenames[i] + '.png')
      #END MH
```

You should change those extensions to match the extensions of your own images if they differ.



Now you can run the file (from the `datasets` directory:

```bash
python3 build_pqr_data.py
```

Once this is done, you will have a `tfrecord` directory filled with `.tfrecord` files.



**Add the information about your dataset segmentation** (TODO: check to make sure we still need this step...)



You'll need to provide tensorflow the list of how your dataset was divided up into training and test images. 

In `deprecated/segmentation_dataset.py` , look for the following (~Line 114):

```
# MH
_PQR_INFORMATION = DatasetDescriptor(
splits_to_sizes={
  'train': 650,
  'val': 150,
  'trainval': 800,
},
num_classes=2,
ignore_label=255,
)

_DATASETS_INFORMATION = {
'cityscapes': _CITYSCAPES_INFORMATION,
'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
'ade20k': _ADE20K_INFORMATION,
'pqr': _PQR_INFORMATION,
}
# END MH
```

These splits should match the number of files in your training and test sets that you made earlier. For example, if `train.txt` has 487 line numbers, `train` is 487. Same with `val` and `trainval`. If you are trying to segment more than just the background and foreground, `num_classes` should match the number of segmentations you are targeting. `ignore_label=255` just means you are ignoring anything in the segmentation that is white (used in some segmentations to create a clear space division between multiple segmentations).



Note that `_DATASETS_INFORMATION` also contains a reference to this new dataset descriptor we've added:

`'pqr': _PQR_INFORMATION`



You're finally ready to train!



### Training Process



**Folder Structure**

Make sure your folder structure from `/datasets` looks similar to this, if you followed all of the naming conventions in the above steps:


```
+ PQR
  + exp //contains exported files
  + train_on_trainval_set
  + eval //contains results of training evaluation
  + init_models //contains the deeplab pascal training set, which you need to download
  + train //contains training ckpt files
  + vis
    + segmentation_results //contains the generated segmentation masks
  + Imagesets
    train.txt
    trainval.txt
    val.txt
  + logs
  + tfrecord //holds your converted dataset
buid_pqr_data.py //creates your tfrecord files
convert_rgb_to_index.py //turns rgb images into their segmentation indices

../../train-pqr.sh //holds the training script
../../eval-pqr.sh //holds the eval script
../../vis-pqr.sh //holds the visualization script
```



**Download the Pascal Training Set**

In order to make our training *much* faster we'll want to use a pre-trained model, in this case pascal VOC2012. [You can download it here](http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz). Extract it into the `PQR/exp/train_on_tranval_set/init_models` directory (should be named `deeplabv3_pascal_train_aug`).









**Edit your training script**

First, edit your `train-pqr.sh` script (in the `models/research`) directory:

```bash
# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
PQR_FOLDER="PQR"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/${EXP_FOLDER}/train"
DATASET="${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/tfrecord"

mkdir -p "${WORK_DIR}/${DATASET_DIR}/${PQR_FOLDER}/exp"
mkdir -p "${TRAIN_LOGDIR}"

NUM_ITERATIONS=9000
python3 "${WORK_DIR}"/train.py \
--logtostderr \
--train_split="train" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--train_crop_size=1000,667 \
--train_batch_size=4 \
--training_number_of_steps="${NUM_ITERATIONS}" \
--fine_tune_batch_norm=true \
--tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
--train_logdir="${TRAIN_LOGDIR}" \
--dataset_dir="${DATASET}"
```

Things you may want to change:

* Make sure all paths are correct (starting from th `models/research` folder as `CURRENT_DIR`)
* `NUM_ITERATIONS` - this is how long you want to train for. For me, on a Macbook Pro without GPU support, it took about 12 hours just to run 1000 iterations. You can expect GPU support to speed that up about 10X. At 1000 iterations, I still had a loss of about `.17`. I would recommend at least 3000 iterations. Some models can be as high as about 20000. You don't want to overtrain, but you're better off over-training than under-training. (For now, I have only done 1000, as this mac is rubbish and internet is not reliable to connect remotely)
* `train_cropsize` - this is the size of the images you are training on. Your training will go **much** faster on smaller images. 1000x667 is quite large and I'd have done better to reduce that size a bit before training. Also, you should make sure these dimensions match in all three scripts: `train-pqr`,`eval-pqr`, and `vis-pqr.py`. #TODO - find out if 513 is important here....
* The checkpoint files (`.ckpt`) are stored in your `PQR_FOLDER` and can be quite large (mine were 330 MB per file). However, periodically (in this case every 4 checkpoint files), the oldest checkpoint file will be deleted and the new one added - this should keep your harddrive from filling up too much. But in general, make sure you have plenty of harddrive space.







**Start training**:

You are finally ready to start training!

From the `models/research` directory, run `sh train-pqr.sh`

If you've set everything up properly, your machine should start training! This will take.a.long.time. You should be seeing something like this in your terminal:



![training](https://prototypes.mikeheavers.com/transfer/deeplab/readme_images/training.png)



**Evaluation**

Running `eval-pqr.sh` from the same directory will calculate the [`mean intersection over union`](https://www.jeremyjordan.me/evaluating-image-segmentation-models/) score for your model. Essentially, this will tell you the number of pixels in common between the actual mask and the prediction of your model


In my case, I got a score of ~`.87` - which means essentially 87% of the pixels in my prediction mask were found in my target mask. The higher the number here, the better the mask.
TODO - this is currently not working for me


**Visualization**

To visualize the actual output of your masks, run `vis-pqr.sh` from the `models/research` directory. These will output to your visualization directory you specified (in our case, `models/research/deeplab/datasets/PQR/exp/train_on_trainval_set/vis/segmentation_results`).  You will see two separate images for each visualization: the "regular" image, and the "prediction" (or segmentation mask). 
TODO - also currently not working
