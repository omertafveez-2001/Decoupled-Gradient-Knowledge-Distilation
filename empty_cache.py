import torch

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared!")
    else:
        print("CUDA is not available. No action taken.")

clear_gpu_cache()