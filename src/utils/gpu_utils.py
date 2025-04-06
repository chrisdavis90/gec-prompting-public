import logging
from pynvml import *

logger = logging.getLogger(__name__)


def print_gpu_utilization():
    logger.info("GPU memory utilization:")
    nvmlInit()
    n_devices = nvmlDeviceGetCount()

    memory_used = {}
    for i in range(n_devices):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        memory_used[i] = info.used // 1024**2
        logger.info(f"GPU memory {i} occupied: {memory_used[i]} MB.")
