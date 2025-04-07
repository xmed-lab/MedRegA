#!/bin/bash

#SBATCH -J tst_gd_l #Slurm job name
#SBATCH --time=3-00:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mail-user=lwangdk@connect.ust.hk   # email address
#SBATCH --mail-type=FAIL

module swap cuda11.8/toolkit/11.8.0
set -x

GPUS=${GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
META_FILE='meta_file.json'  ## all the training data here

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34116
export LAUNCHER=pytorch
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='medrega'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

srun --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  ${SRUN_ARGS} \
  python -u internvl/train/internvl_chat_finetune.py \
  --model_name_or_path 'OpenGVLab/InternVL-Chat-V1-2' \
  --conv_style "Hermes-2" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path ${META_FILE} \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp True \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 2 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1300 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --group_by_length True \
  --do_train True \
  --grad_checkpoint True \
  --deepspeed "zero_stage3_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
