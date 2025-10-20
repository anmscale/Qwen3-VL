#!/bin/bash


# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=4

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-7B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=5e-5
batch_size=4
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py


# Dataset configuration
datasets=mscoco2017_train_captions

# Output configuration
run_name="qwen2vl-coco"
output_dir=/mnt/local_storage/qwen-vl-finetune/checkpoints/

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --seed 42 \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --max_steps 100 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy no \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type constant \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --run_name ${run_name} \
    --report_to none"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        ${entry_file} ${args}
