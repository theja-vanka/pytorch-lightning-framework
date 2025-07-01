export CONFIG="conf/models/model_4c.yaml"
export CHECKPOINT="/mnt/home/dev/usr/tffc-vol-2/krishnatheja/road_visibility_multi_label/lightning_classifier/artifacts/iter_20250208_055302_class3/RC_20250208_55754_ep004.ckpt"
export BATCHSIZE=1536 # 1024 # 1536
export CSVFILE="/mnt/home/dev/usr/tffc-vol-2/krishnatheja/road_visibility_multi_label/lightning_classifier/test_on_unlabelled.csv"
export DEVRUN=0 # Set to 1 to test run

COMMAND="CUDA_VISIBLE_DEVICES=0 python inference_unlabelled.py \
--model_checkpoint $CHECKPOINT \
--batchsize $BATCHSIZE \
--csvfile $CSVFILE \
--fast_dev_run $DEVRUN \
--config $CONFIG"

# run training
echo $COMMAND
eval $COMMAND