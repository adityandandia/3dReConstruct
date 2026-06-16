import torch
import sys
import subprocess
import os

def check_env():
    print("--- Environment Check ---")
    
    # 1. Python & PyTorch
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # 2. CUDA Compatibility
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    # 3. Compiler Check
    try:
        nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode()
        print(f"NVCC Found: Yes")
        # Extract version
        import re
        version = re.search(r"release (\d+\.\d+)", nvcc_output)
        print(f"NVCC Version: {version.group(1) if version else 'Unknown'}")
    except Exception:
        print("NVCC Found: No (Need CUDA Toolkit)")

    # 4. GCC Check
    try:
        gcc_output = subprocess.check_output(["gcc", "--version"]).decode().split('\n')[0]
        print(f"GCC Version: {gcc_output}")
    except Exception:
        print("GCC Found: No")

    # 5. Build Capability Check
    print("\n--- Compatibility Assessment ---")
    if not cuda_available:
        print("CRITICAL: CUDA is not available. 3DGS training requires an NVIDIA GPU with drivers.")
    
    if torch.cuda.get_device_capability(0) < (8, 0):
        print("WARNING: GPU compute capability < 8.0. FastGS may be slow or incompatible.")
    
    if torch.version.cuda and version:
        if float(torch.version.cuda) > float(version.group(1)):
            print("CRITICAL: PyTorch CUDA version is newer than NVCC version. This causes build errors.")
        else:
            print("Status: Versions look compatible for compilation.")

if __name__ == "__main__":
    check_env()
