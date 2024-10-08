:<<! 
[script description]: build a elasticsearch database for corpus
[dataset]: multi domain DE-EN dataset
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE
dataset=$1
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
DATA_PATH=$PROJECT_PATH/data-bin/$dataset
ELASTIC_INDEX_NAME=$dataset

python $PROJECT_PATH/knnbox-scripts/simple-scalable-knn-mt/create_elasticsearch_index.py \
--dataset-path $DATA_PATH \
--elastic-index-name $ELASTIC_INDEX_NAME \


