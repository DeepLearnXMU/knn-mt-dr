export OMP_WAIT_POLICY=PASSIVE
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..

export CUDA_VISIBLE_DEVICES=0
ARCH='base'
DATASET='it'
knn_max_k=8
DATASTORE_PATH=$PROJECT_PATH/datastore/vanilla/$DATASET
DATA_PATH=$PROJECT_PATH/datastore/me/$DATASET
SAVE_PATH=$PROJECT_PATH/save-models/skip-combiner/$ARCH/$DATASET
BASE_MODEL_PATH=$PROJECT_PATH/pretrain-models/wmt19.de-en

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
*)
    python $PROJECT_PATH/knnbox-scripts/skip-knn-mt/train.py \
        --arch mev3 \
        --dataset $DATASET \
        --datastore-path $DATASTORE_PATH \
        --data-path $DATA_PATH \
        --save-path $SAVE_PATH \
        --base-model-path $BASE_MODEL_PATH \
        --knn-max-k $knn_max_k \
        --batch-size 2048 \
        --max-epoch 500 \
        --patience 30 \
        --seed 1 \
        --move-data-to-gpu \
        --alpha-coef 2.0 \
        --alpha-mode v1 \
        --temperature $temperature \
        --knn-lambda $lambda \
        #--with-adaptive
	# $(if [ "$ARCH" == "adaptive" ]; then echo "--with-adaptive"; fi)
    ;;
esac
