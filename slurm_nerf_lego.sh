#!/bin/bash
# The interpreter used to execute the script
#\#SBATCH" directives that convey submission options:
#SBATCH --job-name=nerf_lego
# For email notifications, pass --mail-user on the sbatch command line.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=nerf_lego_%j.log
#SBATCH --error=nerf_lego_%j.err 
#SBATCH --partition=gpu_mig40,gpu,spgpu
#SBATCH --gres=gpu:1
echo "hello world"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nerf

python unit_test.py --test get_rays
python unit_test.py --test compute_bin_dists
python unit_test.py --test compute_alpha
python unit_test.py --test compute_weights

echo "============test done=================="

python run_nerf.py --config configs/lego_fine.txt

echo "done"
