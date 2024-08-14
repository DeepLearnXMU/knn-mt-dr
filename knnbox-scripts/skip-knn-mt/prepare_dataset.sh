# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE
export CUDA_VISIBLE_DEVICES=0

dataset="it"
max_k=8
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
DATASTORE_PATH=$PROJECT_PATH/datastore/vanilla/${dataset}
SAVE_PATH=$PROJECT_PATH/datastore/me/${dataset}
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en

mkdir -p $SAVE_PATH

echo "prepare valid dataset for skip-knn-mt"
python $PROJECT_PATH/knnbox-scripts/skip-knn-mt/prepare_valid_dataset.py \
    --dataset $dataset \
    --datastore-path $DATASTORE_PATH \
    --save-path $SAVE_PATH \
    --model-path $BASE_MODEL \
    --knn-k $max_k \

