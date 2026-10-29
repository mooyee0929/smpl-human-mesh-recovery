#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --output=gpu_check_%j.log
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=4G

echo "========== Node Info =========="
hostname
date

echo -e "\n========== GPU Hardware Check =========="
nvidia-smi

echo -e "\n========== CUDA Environment =========="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo -e "\n========== PyTorch GPU Check =========="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 4Dhumans

python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
    
    # 測試 GPU 運算
    print("\n========== GPU Compute Test ==========")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print("✓ GPU compute test passed!")
else:
    print("\n⚠ WARNING: No GPU detected by PyTorch!")
EOF

echo -e "\n========== Done =========="
date