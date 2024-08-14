:<<! 
[script description]: build a datastore for vanilla-knn-mt
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

dataset=$1
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
#BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
BASE_MODEL=$PROJECT_PATH/pretrain-models/export/cwmt17_zhen.base.pt
DATA_PATH=$PROJECT_PATH/data-bin/${dataset}
DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla/${dataset}/valid

CUDA_VISIBLE_DEVICES=3 python $PROJECT_PATH/knnbox-scripts/common/validate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--dataset-impl mmap \
--valid-subset valid \
--skip-invalid-size-inputs-valid-test \
--batch-size 128 \
--quiet \
--bpe fastbpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch vanilla_knn_mt@transformer_wmt19_de_en \
--knn-mode build_datastore \
--knn-datastore-path $DATASTORE_SAVE_PATH \
