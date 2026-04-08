#!/bin/bash
#SBATCH --job-name=ssl_finetune
#SBATCH --output=/home/alkhudmk/opensportslib_ssl/opensportslib/slurm_logs/ssl_finetune_%j.out
#SBATCH --error=/home/alkhudmk/opensportslib_ssl/opensportslib/slurm_logs/ssl_finetune_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

module load python/3.11.0
module load cuda/12.2

export LD_LIBRARY_PATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="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"
export PYTHONPATH="/home/alkhudmk/opensportslib_ssl/opensportslib:${PYTHONPATH}"

PROJECT_DIR="/home/alkhudmk/opensportslib_ssl/opensportslib"
VENV_DIR="/ibex/scratch/alkhudmk/opensportslib_venv"
PYTHON="${VENV_DIR}/bin/python"

cd /home/alkhudmk

echo "=========================================="
echo "SSL Fine-tuning: MAE encoder → Classification"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

${PYTHON} -u << 'PYEOF'
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

from opensportslib import model

api = model.classification(
    config='/home/alkhudmk/opensportslib_ssl/opensportslib/opensportslib/config/sngar_frames_ssl_finetune.yaml',
    data_dir='/ibex/scratch/alkhudmk/sngar-frames',
    save_dir='/ibex/scratch/alkhudmk/ssl_finetune_checkpoints'
)

best_ckpt = api.train(
    train_set='/ibex/scratch/alkhudmk/sngar-frames/annotations_train.json',
    valid_set='/ibex/scratch/alkhudmk/sngar-frames/annotations_valid.json',
    use_ddp=False,
    use_wandb=True
)

print(f'Best checkpoint: {best_ckpt}')

metrics = api.infer(
    test_set='/ibex/scratch/alkhudmk/sngar-frames/annotations_test.json',
    use_wandb=True
)

print(f'Test metrics: {metrics}')
PYEOF

echo ""
echo "End time: $(date)"
echo "DONE"
