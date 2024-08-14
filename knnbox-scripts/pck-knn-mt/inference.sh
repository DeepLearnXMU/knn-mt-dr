:<<!
[script description]: use pck-mt datastore and adaptive combiner to translate 
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-ENscript
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE
dataset=$1
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/$dataset
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/pck/$dataset'_dim64'
COMBINER_LOAD_DIR=$PROJECT_PATH/save-models/combiner/pck/$dataset'_dim64'
MAX_K=8


RESULT_DIR=results/$dataset
CUDA_VISIBLE_DEVICES=7 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset test \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--batch-size 128 \
--scoring sacrebleu \
--tokenizer moses --remove-bpe \
--arch pck_knn_mt@transformer_wmt19_de_en \
--user-dir $PROJECT_PATH/knnbox/models \
--knn-mode inference \
--knn-datastore-path  $DATASTORE_LOAD_PATH \
--knn-max-k $MAX_K \
--knn-combiner-path $COMBINER_LOAD_DIR > $RESULT_DIR/tmp
#--quiet
#--max-tokens 2048 \

cat $RESULT_DIR/tmp | grep -P "^T" | cut -f 2 > $RESULT_DIR/ref                                                                      
cat $RESULT_DIR/tmp | grep -P "^S" | cut -f 2 > $RESULT_DIR/src
cat $RESULT_DIR/tmp | grep -P "^D" | cut -f 3 > $RESULT_DIR/hyp
rm $RESULT_DIR/tmp
