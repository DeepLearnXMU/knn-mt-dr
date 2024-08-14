:<<! 
[script description]: save PLAC drop index given k_p
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE
export CUDA_VISIBLE_DEVICES=$1
domain=$2
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
PLAC_DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/plac/${domain}'_temp'
PLAC_K=8
PLAC_BSZ=4096

python $PROJECT_PATH/knnbox-scripts/plac-knn-mt/save_drop_index.py \
    --plac-datastore-path $PLAC_DATASTORE_SAVE_PATH \
    --plac-k $PLAC_K \
    --plac-bsz $PLAC_BSZ
