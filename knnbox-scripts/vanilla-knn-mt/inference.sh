:<<! 
[script description]: use vanilla-knn-mt to translate
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line will speed up faiss
export OMP_WAIT_POLICY=PASSIVE
export CUDA_VISIBLE_DEVICES=$1
dataset=$2
batch=$3
step=$4
k=8
echo "vanilla-knn-mt "$dataset" batch="$batch 

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
#BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
BASE_MODEL=$PROJECT_PATH/pretrain-models/export/cwmt17_zhen.base.pt
#DATA_PATH=$PROJECT_PATH/data-bin/${dataset}
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/${dataset}
#DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/all_domain
#DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/'it_plac_k8_ratio0.40'

RESULT_DIR=results/${dataset}
if [ ! -d results ]; then mkdir results; fi
if [ ! -d "$RESULT_DIR" ]; then mkdir $RESULT_DIR; fi

case $dataset in
koran)
    temperature=100
    ;;
*)
    temperature=10
    ;;
esac
case $dataset in
it|subtitles)
    lambda=0.7
    ;;
*)
    lambda=0.8
    ;;
esac

echo $temperature
echo $lambda

python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset test \
--batch-size $batch \
--scoring sacrebleu \
--tokenizer moses \
--remove-bpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch vanilla_knn_mt@transformer_wmt19_de_en \
--knn-mode inference \
--knn-datastore-path $DATASTORE_LOAD_PATH \
--knn-k $k \
--knn-lambda $lambda \
--knn-temperature $temperature > $RESULT_DIR/tmp
#--quiet

cat $RESULT_DIR/tmp | grep -P "^T" | cut -f 2 > $RESULT_DIR/ref
cat $RESULT_DIR/tmp | grep -P "^S" | cut -f 2 > $RESULT_DIR/src
#cat $RESULT_DIR/tmp | grep -P "^H" | cut -f 3 > $RESULT_DIR/hyp_tok.$step
cat $RESULT_DIR/tmp | grep -P "^D" | cut -f 3 > $RESULT_DIR/hyp
#cat $RESULT_DIR/tmp | grep -P "^D" | cut -f 3 > $RESULT_DIR/hyp.vanilla
rm $RESULT_DIR/tmp

#CUDA_VISIBLE_DEVICES=1 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
#--task translation \
#--path $BASE_MODEL \
#--dataset-impl mmap \
#--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
#--gen-subset test \
#--max-tokens 16384 \
#--quiet \
#--scoring sacrebleu \
#--tokenizer moses \
#--remove-bpe \
#--user-dir $PROJECT_PATH/knnbox/models \
#--arch vanilla_knn_mt@transformer_wmt19_de_en \
#--knn-mode inference \
#--knn-datastore-path $DATASTORE_LOAD_PATH \
#--knn-k $k \
#--knn-lambda $lambda \
#--knn-temperature $temperature
