#!/bin/bash
#SBATCH --job-name=llama3-self-rag
#SBATCH --output=/home/wberriosr/slurm_logs/p72_e5mistral__%A_%a.out
#SBATCH --nodes=1
#SBATCH --partition=a3high
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus=1
#SBATCH --mem=1500G
#SBATCH --time=10000:00:00

set -eo pipefail
set -x

ulimit -Sn 20480


source /env/bin/start-ctx-user
conda activate self-grit

python -c 'import torch; from packaging import version; assert version.parse(torch.__version__) >= version.parse("2.2.0"), f"TCPx requires torch>=-2.2 - update your conda"'

#ACCELERATE_CONFIG_FILE=accelerate_deepspeed.yaml
ACCELERATE_CONFIG_FILE=accelerate_singlegpu.yaml
GPUS_PER_NODE=1
NNODES=$SLURM_NNODES
NUM_GPUS_TOTAL=$((GPUS_PER_NODE*SLURM_NNODES))
echo $NUM_GPUS_TOTAL

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "10000" "30000" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
export MASTER_PORT=$(unused_port)

LAUNCHER="python -u -m accelerate.commands.launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_machines $SLURM_NNODES \
    --num_processes $NUM_GPUS_TOTAL \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): --tee 3 \
    "

PROGRAM="\
../retrieval_lm/finetune.py \
"
export CMD="$LAUNCHER $PROGRAM"
echo $CMD


SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$CMD"