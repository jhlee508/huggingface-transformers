#!/bin/bash -l

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpt2

DISTRIBUTED_ARGS="--nnodes=2 --nproc_per_node=4 --node-rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=192.168.0.130:1234"

#NCCL_DEBUG=INFO \
#NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL \
# accelerate launch \
#     --config_file /home/n4/jaehwan/.cache/huggingface/accelerate/a7_config.yaml \
#python ../run_clm.py \
#MASTER_ADDR=b0 \

BATCH_SIZE=4

OMP_NUM_THREADS=1 \
NCCL_IB_GID_INDEX=3 \
torchrun $DISTRIBUTED_ARGS ../run_clm.py \
    --model_name_or_path gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --save_steps=500 \
    --output_dir /data/s0/jaehwan/gpt_train_logs \
    --overwrite_output_dir \
    --gradient_checkpointing \
    --deepspeed ./ds_zero3.json
