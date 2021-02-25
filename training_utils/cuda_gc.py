import gc
import torch

import settings
from utils.stopwatch import Stopwatch


def collect():
    """
    Forces gc to dispose already unused python objects that can still
    hold references to cuda tensors allocated on the GPU, freeing up
    the memory. Keep in mind that gc.collect is really slow function
    (3-500ms), so it shouldn't be abused.
    """
    stopwatch = Stopwatch("CUDA_GC")
    with stopwatch("Collecting garbage", color='yellow'):
        initial = torch.cuda.memory_allocated(settings.device)
        gc.collect()
        torch.cuda.empty_cache()
        stopwatch.log(
            f"{settings.device} memory: {initial//1024**2} MB -> "
            f"{torch.cuda.memory_allocated(settings.device)//1024**2} MB",
            "yellow")

# todo: wrap Cuda OOM errors with cuda_gc.collect to preven
