# From tensorflow/models/research/
# Assume all checkpoint files share the same path prefix `${CHECKPOINT_PATH}`.
# CHECKPOINT_PATH = "/deeplab/datasets/PQR/exp/train_on_trainval_set/train/model.ckpt-1010.data-00000-of-00001"
# OUTPUT_DIR = "/deeplab/datasets/PQR/exp/train_on_trainval_set/trained_model"
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"


python deeplab/export_model.py \
    --model_variant="xception_65" \
    --checkpoint_path="${WORK_DIR}/datasets/PQR/exp/train_on_trainval_set/train/model.ckpt-200index" \
    --export_path="${WORK_DIR}/datasets/PQR/exp/train_on_trainval_set/trained_model" \
    --num_classes=2 \
    --crop_size=513 \
    --crop_size=513 