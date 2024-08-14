export OMP_WAIT_POLICY=PASSIVE
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..

export CUDA_VISIBLE_DEVICES=$1

ARCH='base'
DATASET=$2
#DATASTORE_PATH=$PROJECT_PATH/datastore/vanilla/$DATASET
DATASTORE_PATH=$PROJECT_PATH/datastore/vanilla/all_domain
DATA_PATH=$PROJECT_PATH/datastore/me/$DATASET'_all_domain'
SAVE_PATH=$PROJECT_PATH/save-models/faster-combiner/$ARCH/$DATASET'_all_domain'
BASE_MODEL_PATH=$PROJECT_PATH/pretrain-models/wmt19.de-en
#BASE_MODEL_PATH=$PROJECT_PATH/pretrain-models/export

mkdir -p $SAVE_PATH

case $DATASET in
koran)
    temperature=100
    ;;
*)
    temperature=10
    ;;
esac
case $DATASET in
it|subtitles)
    lambda=0.7
    ;;
*)
    lambda=0.8
    ;;
esac

echo $temperature
echo $lambda

case $ARCH in
base|adaptive)
    python $PROJECT_PATH/knnbox-scripts/faster-knn-mt/train.py \
        --arch mev3 \
        --dataset $DATASET \
        --datastore-path $DATASTORE_PATH \
        --data-path $DATA_PATH \
        --save-path $SAVE_PATH \
        --base-model-path $BASE_MODEL_PATH \
        --knn-max-k 8 \
        --batch-size 2048 \
        --max-epoch 100 \
        --patience 100 \
        --seed 3 \
        --move-data-to-gpu \
        --temperature $temperature \
        --knn-lambda $lambda \
        $(if [ "$ARCH" == "adaptive" ]; then echo "--with-adaptive"; fi)
    ;;
*)
    echo "unknown arch"
    ;;
esac
