:<<!
    Train the dimension redcution network.
    we'll use the trained network in two places:
    1. when executing [bash reduct_dimension.sh], we use the trained network reducting the datastore's dimension
    2. when inference, we use this network reducting the dimension of NMT hidden state (a.k.a query)
!

export OMP_WAIT_POLICY=PASSIVE
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
export CUDA_VISIBLE_DEVICES=$1
domain=$2
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/pck/$domain  # the train set is constructed from datastore
LOG_PATH=$DATASTORE_LOAD_PATH/train_log
REDUCT_DIM=64
DR_LOSS_RATIO=0.0
NCE_LOSS_RATIO=1.0
WP_LOSS_RATIO=0.0
BATCH_SIZE=4096
case $domain in
koran)
    MAX_UPDATE=50000
    ;;
it)
    MAX_UPDATE=100000
    ;;
medical)
    MAX_UPDATE=150000
    ;;
law|subtitles)
    MAX_UPDATE=300000
    ;;
esac
#MAX_UPDATE=50000   # Koran: 50000 IT: 100000 Medical: 150000 Law: 300000

python $PROJECT_PATH/knnbox-scripts/pck-knn-mt/train_reduct_network.py \
--datastore-load-path $DATASTORE_LOAD_PATH \
--log-path $LOG_PATH \
--reduct-dim $REDUCT_DIM \
--dataset-sample-rate 1.0 \
--batch-size $BATCH_SIZE \
--learning-rate 3e-4 \
--min-learning-rate 3e-5 \
--patience 30 \
--valid-interval 1000 \
--dr-loss-ratio $DR_LOSS_RATIO \
--nce-loss-ratio $NCE_LOSS_RATIO \
--wp-loss-ratio $WP_LOSS_RATIO \
--max-update $MAX_UPDATE \
