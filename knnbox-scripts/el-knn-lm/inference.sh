:<<!
[script description]: use robust-knn-mt to translate 
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-ENscript
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE
export CUDA_VISIBLE_DEVICES=$1
dataset=$2
bsz=$3
k=8
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/${dataset}
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/${dataset}
COMBINER_LOAD_DIR=$PROJECT_PATH/datastore/el-knnlm/${dataset}/checkpoint_best.pt
#ADAPTIVE_COMBINER_LOAD_DIR=$PROJECT_PATH/save-models/combiner/adaptive/${dataset}
FREQ_CACHE_PATH=$PROJECT_PATH/datastore/el-knnlm/${dataset}/freq_train_cnt.pickle
FERT_CACHE_PATH=$PROJECT_PATH/datastore/el-knnlm/${dataset}/fert_train_cnt.pickle

case $dataset in
koran)
    temperature=100
    ;;
it)
    temperature=10
    ;;
law)
    temperature=10
    ;;
medical)
    temperature=10
    ;;
subtitles)
    temperature=10
    ;;
*)
    temperature=100
    ;;
esac


python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset test \
--batch-size $bsz \
--scoring sacrebleu \
--tokenizer moses \
--remove-bpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch el_knn_mt@transformer_wmt19_de_en \
--knn-mode inference \
--knn-datastore-path $DATASTORE_LOAD_PATH \
--knn-k $k \
--knn-temperature $temperature \
--knn-combiner-path $COMBINER_LOAD_DIR \
--is-el-knnmt \
--freq-path $FREQ_CACHE_PATH \
--fert-path $FERT_CACHE_PATH \
--threshold 0.75 \
--quiet
#--with-adaptive \
#--adaptive-combiner-path $ADAPTIVE_COMBINER_LOAD_DIR
