import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the CUDA version
    cuda_version = torch.version.cuda

    # Get the CUDA root directory
    cuda_home = torch._C._cuda_getCompiledVersion()

    print("CUDA version:", cuda_version)
    print("CUDA root directory:", cuda_home)
else:
    print("CUDA is not available.")