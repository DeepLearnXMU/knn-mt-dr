:<<!
[script description]: use robust-knn-mt to translate 
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-ENscript
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

dataset=$1
batch=$2
echo "robust-knn-mt "$dataset" batch="$batch 


PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/${dataset}
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/${dataset}
COMBINER_LOAD_DIR=$PROJECT_PATH/save-models/combiner/robust/${dataset}
MAX_K=8

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset test \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--batch-size $batch \
--quiet \
--scoring sacrebleu \
--tokenizer moses --remove-bpe \
--arch robust_knn_mt@transformer_wmt19_de_en \
--user-dir $PROJECT_PATH/knnbox/models \
--knn-mode inference \
--knn-datastore-path  $DATASTORE_LOAD_PATH \
--knn-max-k $MAX_K \
--knn-combiner-path $COMBINER_LOAD_DIR \
--skip \
--skip-threshold 0.15