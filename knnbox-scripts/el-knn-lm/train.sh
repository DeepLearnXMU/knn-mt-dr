# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE
export CUDA_VISIBLE_DEVICES=$1
dataset=$2

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
EL_DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/el-knnlm/${dataset}

python $PROJECT_PATH/knnbox-scripts/el-knn-lm/train_ar.py \
    --dataset $dataset \
    --seed 1 \
    --lr 0.0005 \
    --l1 0.05 \
    --batch-size 64 \
    --ngram 0 \
    --hidden-units 128 \
    --nlayers 5 \
    --dropout 0.2 \
    --cache-path $EL_DATASTORE_SAVE_PATH 

