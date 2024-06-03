MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=256
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
RUN_NAME="self_rag_${MODEL_SIZE}_8gpus_${TOTAL_BATCH_SIZE}_bs_origin"
OUTPUT_DIR="/data/wberriosr/self-rag-exps/${RUN_NAME}/"
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
export WANDB_NAME=$RUN_NAME

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file stage3_no_offloading_accelerate.conf \
    finetune.py \
    --model_name_or_path /data/models/hf_llama2_7b/ \
    --use_flash_attn \
    --tokenizer_name /data/models/hf_llama2_7b/ \
    --use_slow_tokenizer \
    --dataset_name selfrag/selfrag_train_data \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 4e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --use_special_tokens