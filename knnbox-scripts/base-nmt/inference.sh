:<<!
[script description]: use neural machine translation model to translate 
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss. base nmt dosent need faiss, 
# we set this environment variable here just for fair comparison.
export OMP_WAIT_POLICY=PASSIVE
export CUDA_VISIBLE_DEVICES=$1
domain=$2
batch=$3
echo "base-nmt "$dataset" batch="$batch 

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
DATA_PATH=$PROJECT_PATH/data-bin/${domain}
#BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
BASE_MODEL=$PROJECT_PATH/pretrain-models/export/cwmt17_zhen.base.pt

RESULT_DIR=results/${domain}
if [ ! -d 'results' ]; then mkdir 'results'; fi
if [ ! -d "$RESULT_DIR" ]; then mkdir $RESULT_DIR; fi

python $PROJECT_PATH/fairseq_cli/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang zh --target-lang en \
--gen-subset test \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--batch-size $batch \
--scoring sacrebleu \
--tokenizer moses --remove-bpe > $RESULT_DIR/tmp

cat $RESULT_DIR/tmp | grep -P "^T" | cut -f 2 > $RESULT_DIR/ref
cat $RESULT_DIR/tmp | grep -P "^S" | cut -f 2 > $RESULT_DIR/src
cat $RESULT_DIR/tmp | grep -P "^D" | cut -f 3 > $RESULT_DIR/hyp
rm $RESULT_DIR/tmp
