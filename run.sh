data_dir=./data
model_dir=./weights

python -u -m torch.distributed.launch --nproc_per_node=4 train.py \
    --kernel-type rex20_final_256_400w_f4_10ep \
    --train-step 0 \
    --data-dir ${data_dir} \
    --image-size 256 \
    --batch-size 64 \
    --enet-type rex20 \
    --n-epochs 10 \
    --fold 4 \
    --CUDA_VISIBLE_DEVICES 0,1,2,3
    