:<<!
    reduct the datastore dimension using the trained network
!

export OMP_WAIT_POLICY=PASSIVE
export CUDA_VISIBLE_DEVICES=$1
dataset=$2
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/pck/$dataset/valid
OUTPUT_DATASTORE_PATH=$PROJECT_PATH/datastore/pck/$dataset'_dim64'/valid

python $PROJECT_PATH/knnbox-scripts/pck-knn-mt/reduct_datastore_dim.py \
--input-datastore-path $DATASTORE_LOAD_PATH \
--output-datastore-path $OUTPUT_DATASTORE_PATH \
