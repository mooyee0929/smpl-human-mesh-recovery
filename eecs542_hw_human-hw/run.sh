#!/bin/bash
#SBATCH --job-name=fsyang_test
#SBATCH --mail-user=fsyang@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=fsyang_test%j.log
#SBATCH --error=fsyang_test%j.err
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "==================== Job Info ===================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo ""

echo "==================== GPU Check ===================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo ""

echo "==================== PyTorch GPU Check ===================="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 4Dhumans

python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')"
echo ""

export HYDRA_FULL_ERROR=1

echo "==================== Running Script ===================="
python track.py video.source="./example_data/videos/example_1/images" \
        video.output_dir=results/example_1/hmr2

python extract_frames.py --video results/example_1/hmr2/PHALP_images.mp4 \
        --out_dir results/example_1/hmr2/extracted_frames

echo -e "\n========== Checking Output Files =========="
find results/example_1/hmr2 -type f
ls -R results/example_1/hmr2/


echo "done"