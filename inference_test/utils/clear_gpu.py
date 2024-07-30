import torch
import gc

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()