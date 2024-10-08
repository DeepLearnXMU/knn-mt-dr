:<<! 
[script description]: use vanilla-knn-mt to translate
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line will speed up faiss
export OMP_WAIT_POLICY=PASSIVE

dataset=$1
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/${dataset}
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/${dataset}

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
esac

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset test \
--batch-size 128 \
--quiet \
--scoring sacrebleu \
--tokenizer moses \
--remove-bpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch vanilla_knn_mt@transformer_wmt19_de_en \
--knn-mode inference \
--knn-datastore-path $DATASTORE_LOAD_PATH \
--knn-k $k \
--knn-lambda $lambda \
--knn-temperature $temperature \