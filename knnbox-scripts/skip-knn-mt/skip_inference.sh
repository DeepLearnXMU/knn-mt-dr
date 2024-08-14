:<<!
[script description]: use robust-knn-mt to translate 
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-ENscript
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

export CUDA_VISIBLE_DEVICES=0
ARCH='base'
DATASET='it'
batch=128

echo $ARCH" "$DATASET

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/$DATASET
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/$DATASET
COMBINER_LOAD_PATH=$PROJECT_PATH/save-models/skip-combiner/$ARCH/$DATASET/checkpoint_best.pt
MAX_K=8

echo $DATASTORE_LOAD_PATH

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
koran|subtitles)
    alphamin=0.45
    ;;
*)
    alphamin=0.4
    ;;
esac

echo $temperature
echo $lambda

datasetlength=$(python length.py $DATASET)

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
    --arch skip_knn_mt@transformer_wmt19_de_en \
    --user-dir $PROJECT_PATH/knnbox/models \
    --knn-mode inference \
    --knn-datastore-path  $DATASTORE_LOAD_PATH \
    --knn-max-k $MAX_K \
    --knn-combiner-path $COMBINER_LOAD_PATH \
    --skip-inference \
    --alphamin $alphamin \
    --datasetlength $datasetlength \
    --knn-lambda $lambda \
    --knn-temperature $temperature \
    --dataset $DATASET \
    --quiet
