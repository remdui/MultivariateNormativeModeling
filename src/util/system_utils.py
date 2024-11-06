"""Utilities for logging system and hardware information."""

import platform

import psutil
import torch

from entities.log_manager import LogManager


def log_system_info() -> None:
    """Log system and hardware information."""

    logger = LogManager.get_logger(__name__)

    # Operating System Information
    os_info = (
        "\nOperating System Details:\n"
        f"  System: {platform.system()}\n"
        f"  Node Name: {platform.node()}\n"
        f"  Release: {platform.release()}\n"
        f"  Version: {platform.version()}\n"
        f"  Machine: {platform.machine()}\n"
        f"  Processor: {platform.processor()}"
    )

    # Hardware Information
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    total_memory = psutil.virtual_memory().total / (1024**3)
    hardware_info = (
        "Hardware Information:\n"
        f"  CPU Count: {cpu_count}\n"
        f"  Total Memory: {total_memory:.2f} GB"
    )
    if cpu_freq:
        hardware_info += f"\n  CPU Frequency: {cpu_freq.current:.2f} MHz"

    # GPU Information
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_info = "CUDA is available. GPU Details:\n"
        gpu_info += f"  Number of GPUs: {num_gpus}\n"
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_capability = torch.cuda.get_device_capability(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_info += (
                f"  GPU {i}:\n"
                f"    Name: {gpu_name}\n"
                f"    Compute Capability: {gpu_capability}\n"
                f"    Total Memory: {gpu_memory:.2f} GB"
            )
    else:
        gpu_info = "CUDA is not available. No GPU found."

    # Python and PyTorch Versions
    version_info = (
        f"Python Version: {platform.python_version()}\n"
        f"PyTorch Version: {torch.__version__}"
    )

    # Combine all the information
    system_info = f"{os_info}\n\n{hardware_info}\n\n{gpu_info}\n\n{version_info}"

    # Log the combined system information
    logger.debug(system_info)
