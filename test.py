from utils.utils import get_best_device

def get_gpu_free_memory_mb(gpu_id: int) -> tuple[float, float]:
    """返回 (free_mb, total_mb)，基于整卡 NVML 统计。"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = mem.free / (1024 ** 2)
        total_mb = mem.total / (1024 ** 2)
        return free_mb, total_mb
    except Exception:
        # 回退：nvidia-smi
        import subprocess
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            text=True,
        )
        free_mb, total_mb = map(float, out.strip().split(","))
        return free_mb, total_mb

device = get_best_device()
print(get_gpu_free_memory_mb(device.index))