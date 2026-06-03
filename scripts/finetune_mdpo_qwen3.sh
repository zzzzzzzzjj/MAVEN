#!/bin/bash

MODEL_NAME="Qwen3-VL-4B-Instruct"

GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=1
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4


deepspeed --master_port 29501 src/train/train_mdpo.py \
    --dpo_loss "sigmoid" \
    --precompute_ref_log_probs False \
    --beta 0.1 \
    --implicit_beta True \
    --use_liger_loss True \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path valuedata/3861_multilevel.json \
    --image_folder  valuedata/raw_data \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir output/qwen3vl_4b_samdpo \
    --num_train_epochs 4 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 32 * 32)) \
    --image_max_pixels $((1280 * 32 * 32)) \
    --learning_rate 1e-4 \
    --merger_lr 5e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_total_limit 4 \
    --dataloader_num_workers 4 \
    --lora_enable True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_levels 4 \
    --max_grad_norm 1.0 \
    --save_strategy "epoch" \
    --save_steps 50 


    


    
