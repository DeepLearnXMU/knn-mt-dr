:<<!
[script description]: use robust-knn-mt to translate 
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-ENscript
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

export CUDA_VISIBLE_DEVICES=$1
ARCH='base'
DATASET=$2
batch=$3
THRESHOLD=0.5

echo $ARCH" "$DATASET

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
#BASE_MODEL=$PROJECT_PATH/pretrain-models/export/cwmt17_zhen.base.pt
DATA_PATH=$PROJECT_PATH/data-bin/$DATASET
#DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/$DATASET
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/all_domain
COMBINER_LOAD_PATH=$PROJECT_PATH/save-models/faster-combiner/$ARCH/$DATASET'_all_domain'/checkpoint_best.pt
MAX_K=8
RESULT_DIR=results/$DATASET'_all'

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

case $DATASET in
it)
    #THRESHOLD=0.6
    THRESHOLD=0.67
    ;;
koran)
    THRESHOLD=0.43
    ;;
law)
    THRESHOLD=0.58
    ;;
medical)
    THRESHOLD=0.62
    ;;
subtitles)
    #THRESHOLD=0.63
    THRESHOLD=0.8
    ;;
esac

echo $temperature
echo $lambda

if [ ! -d "$RESULT_DIR" ]; then mkdir $RESULT_DIR; fi
# kernprof -l -v
python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
    --seed 1 \
    --task translation \
    --path $BASE_MODEL \
    --dataset-impl mmap \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
    --gen-subset test \
    --model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
    --batch-size $batch \
    --scoring sacrebleu \
    --tokenizer moses --remove-bpe \
    --arch faster_knn_mt@transformer_wmt19_de_en \
    --user-dir $PROJECT_PATH/knnbox/models \
    --knn-mode inference \
    --knn-datastore-path  $DATASTORE_LOAD_PATH \
    --knn-max-k $MAX_K \
    --knn-combiner-path $COMBINER_LOAD_PATH \
    --skip-inference \
    --skip-threshold $THRESHOLD \
    --knn-lambda $lambda \
    --knn-temperature $temperature > $RESULT_DIR/tmp
    #--quiet
    #     > $RESULT_DIR/tmp
cat $RESULT_DIR/tmp | grep -P "^T" | cut -f 2 > $RESULT_DIR/ref
cat $RESULT_DIR/tmp | grep -P "^S" | cut -f 2 > $RESULT_DIR/src
cat $RESULT_DIR/tmp | grep -P "^D" | cut -f 3 > $RESULT_DIR/hyp
rm $RESULT_DIR/tmp


#CUDA_VISIBLE_DEVICES=1 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
#    --seed 1 \
#    --task translation \
#    --path $BASE_MODEL \
#    --dataset-impl mmap \
#    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
#    --gen-subset test \
#    --model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
#    --max-tokens 16384 \
#    --scoring sacrebleu \
#    --tokenizer moses --remove-bpe \
#    --arch skip_knn_mt@transformer_wmt19_de_en \
#    --user-dir $PROJECT_PATH/knnbox/models \
#    --knn-mode inference \
#    --knn-datastore-path  $DATASTORE_LOAD_PATH \
#    --knn-max-k $MAX_K \
#    --knn-combiner-path $COMBINER_LOAD_PATH \
#    --skip-inference \
#    --skip-threshold $THRESHOLD \
#    --quiet
    #     > $RESULT_DIR/tmp.$THRESHOLD



#cat $RESULT_DIR/tmp.$THRESHOLD | grep -P "^T" | cut -f 2 > $RESULT_DIR/ref
#cat $RESULT_DIR/tmp.$THRESHOLD | grep -P "^S" | cut -f 2 > $RESULT_DIR/src
#cat $RESULT_DIR/tmp.$THRESHOLD | grep -P "^T" | cut -f 1 > $RESULT_DIR/id
#cat $RESULT_DIR/tmp.$THRESHOLD | grep -P "^D" | cut -f 3 > $RESULT_DIR/hyp.$THRESHOLD
#cat $RESULT_DIR/tmp.$THRESHOLD | grep -P "^D" | cut -f 4,5 > $RESULT_DIR/skip.$THRESHOLD
#rm $RESULT_DIR/tmp.$THRESHOLD


#sed '$d' hyp.$THRESHOLD > temp && mv temp hyp.$THRESHOLD
#sacrebleu $RESULT_DIR/ref.$THRESHOLD -i $RESULT_DIR/hyp.$THRESHOLD -w 2 -b
