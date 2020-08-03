


# /Users/student/Code/uni/fun-with-deeplab/pascal_voc_seg



# Set up the working environment.
# CURRENT_DIR=$(pwd)
# WORK_DIR="${CURRENT_DIR}/deeplab"
# DATASET_DIR="datasets"


# PATH_TO_INITIAL_CHECKPOINT ="/Users/student/Code/uni/fun-with-deeplab/deeplab/datasets/PQR/exp/train_on_trainval_set/init_models/deeplabv3_pascal_train_aug"
# PATH_TO_TRAIN_DIR = "/Users/student/Code/uni/fun-with-deeplab/pascal_voc_seg/exp/trained_checkpoints"
# DATASET="/Users/student/Code/uni/fun-with-deeplab/pascal_voc_seg/tfrecord"

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=1000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="513,513" \
    --train_batch_size=1 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint="/Users/student/Code/uni/fun-with-deeplab/deeplab/datasets/PQR/exp/train_on_trainval_set/init_models/deeplabv3_pascal_train_aug" \
    --train_logdir="/Users/student/Code/uni/fun-with-deeplab/pascal_voc_seg/exp/trained_checkpoints" \
    --dataset_dir="/Users/student/Code/uni/fun-with-deeplab/pascal_voc_seg/tfrecord"