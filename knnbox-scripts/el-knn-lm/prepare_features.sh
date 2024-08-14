# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

dataset=$1

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..

EL_DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/el-knnlm/${dataset}_all_domain
#DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla/${dataset}
DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla/all_domain
DICTIONARY_PATH=$PROJECT_PATH/pretrain-models/wmt19.de-en/fairseq-vocab.txt
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
#BASE_MODEL=$PROJECT_PATH/pretrain-models/export/cwmt17_zhen.base.pt
#DICTIONARY_PATH=$PROJECT_PATH/pretrain-models/export/fairseq-vocab.txt

mkdir -p $EL_DATASTORE_SAVE_PATH

echo "compute frequency/fertility dictionary"
python $PROJECT_PATH/knnbox-scripts/el-knn-lm/cache_freq_fert.py \
    --dataset $dataset \
    --datastore-path $DATASTORE_SAVE_PATH \
    --cache $EL_DATASTORE_SAVE_PATH \
    --dict-path $DICTIONARY_PATH \
    --csize 1 \

case $dataset in
koran)
    temperature=100
    k=8
    lambda=0.8
    ;;
it)
    temperature=10
    k=8
    lambda=0.7
    ;;
law)
    temperature=10
    k=4
    lambda=0.8
    ;;
medical)
    temperature=10
    k=4
    lambda=0.8
    ;;
subtitles)
    temperature=10
    k=8
    lambda=0.7
    ;;
*)
    temperature=100
    k=8
    lambda=0.8
esac

echo "compute nmt_max / nmt_entropy statistics"
python $PROJECT_PATH/knnbox-scripts/el-knn-lm/precompute_all_features.py \
    --dataset $dataset \
    --datastore-path $DATASTORE_SAVE_PATH \
    --cache $EL_DATASTORE_SAVE_PATH \
    --dict-path $DICTIONARY_PATH \
    --model-path $BASE_MODEL \
    --knn-lambda $lambda \
    --knn-temperature $temperature \
    --knn-k $k \
    --csize 1 \
    --subset valid \
