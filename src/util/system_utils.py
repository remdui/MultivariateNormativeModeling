"""System logging utilities for retrieving OS, hardware, and GPU information."""

import platform

import psutil
import torch

from entities.log_manager import LogManager
from entities.properties import Properties


def log_system_info() -> None:
    """
    Logs system and hardware details, including OS, CPU, memory, and GPU information.

    This function gathers relevant system information and logs it at the DEBUG level.
    """
    logger = LogManager.get_logger(__name__)

    # Gather OS information
    os_info = "\n".join(
        [
            "Operating System Details:",
            f"  System: {platform.system()}",
            f"  Node Name: {platform.node()}",
            f"  Release: {platform.release()}",
            f"  Version: {platform.version()}",
            f"  Machine: {platform.machine()}",
            f"  Processor: {platform.processor()}",
        ]
    )

    # Gather CPU and memory information
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = getattr(psutil.cpu_freq(), "current", "Unknown")  # Avoid NoneType errors
    total_memory = psutil.virtual_memory().total / (1024**3)

    hardware_info = "\n".join(
        [
            "Hardware Information:",
            f"  CPU Count: {cpu_count}",
            f"  CPU Frequency: {cpu_freq} MHz",
            f"  Total Memory: {total_memory:.2f} GB",
        ]
    )

    # Gather GPU information
    gpu_info = "CUDA is not available. No GPU detected."
    if torch.cuda.is_available():
        try:
            num_gpus = torch.cuda.device_count()
            gpu_details = [
                f"  GPU {i}: {torch.cuda.get_device_name(i)}"
                f" (Compute Capability: {torch.cuda.get_device_capability(i)})"
                f" | Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB"
                for i in range(num_gpus)
            ]
            gpu_info = "CUDA is available. GPU Details:\n" + "\n".join(gpu_details)
        except OSError as e:
            logger.warning(f"Error retrieving CUDA details: {e}")

    # Python & PyTorch version info
    version_info = "\n".join(
        [
            f"Python Version: {platform.python_version()}",
            f"PyTorch Version: {torch.__version__}",
        ]
    )

    # Combine and log all system information
    system_info = "\n\n".join([os_info, hardware_info, gpu_info, version_info])
    logger.debug(system_info)


def gpu_supported_by_triton_compiler() -> bool:
    """
    Checks if the system's GPU supports Triton compilation for CUDA.

    Triton compilation is supported on NVIDIA V100, A100, and H100 GPUs.
    This function determines if the current CUDA device meets these requirements.

    Returns:
        bool: True if Triton compilation is supported, False otherwise.
    """
    logger = LogManager.get_logger(__name__)
    properties = Properties.get_instance()

    # Ensure CUDA is available and selected as the primary device
    if not torch.cuda.is_available() or properties.system.device != "cuda":
        logger.info("CUDA is either unavailable or not selected as the compute device.")
        return False

    try:
        device_capability = torch.cuda.get_device_capability()
        supported_capabilities = {(7, 0), (8, 0), (9, 0)}  # V100, A100, H100

        if device_capability in supported_capabilities:
            logger.info("CUDA compilation is supported on this device.")
            return True

        logger.warning(
            f"CUDA compilation is NOT supported on this GPU (Compute Capability: {device_capability}). "
            "Recommended GPUs: NVIDIA V100, A100, or H100."
        )
    except RuntimeError as e:
        logger.error(f"Runtime error when querying CUDA device: {e}")
    except IndexError as e:
        logger.error(f"GPU index out of range: {e}")
    except AttributeError as e:
        logger.error(f"Failed to retrieve CUDA device attributes: {e}")

    return False
