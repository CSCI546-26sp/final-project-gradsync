"""
Utilities for device detection and configuration in distributed pipeline training.
"""

import torch


WIRE_DTYPE = torch.float32  # wire format only; in-memory model stays fp32


def pack_tensor(t: torch.Tensor):
    """
    Convert a tensor to wire bytes using lower precision.
    Returns: (bytes, shape)
    """
    t_wire = t.detach().to(dtype=WIRE_DTYPE).cpu().contiguous()
    return t_wire.numpy().tobytes(), list(t_wire.shape)


def unpack_tensor(payload: bytes, shape, device):
    """
    Convert wire bytes back to fp32 tensor for local PyTorch use.
    """
    t_wire = torch.frombuffer(
        bytearray(payload), dtype=WIRE_DTYPE).reshape(shape).clone()
    return t_wire.to(device=device, dtype=torch.float32)


def detect_device() -> torch.device:
    """
    Detect the optimal device for a given node role in distributed training.

    Args:
        role: The node role - 'head' for the head node, 'tail' for the tail node

    Returns:
        torch.device: The detected device for the given role

    Raises:
        ValueError: If role is not 'head' or 'tail'
    """

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        # Head node: prefer MPS on Apple Silicon, fallback to CPU
        device = "mps"
    else:
        device = "cpu"

    print("Device is:", device)
    return torch.device(device)


def get_device_info(device: torch.device) -> dict:
    """
    Get detailed information about a device for logging/debugging.

    Args:
        device: The torch device to inspect

    Returns:
        dict: Device information including name, type, and capabilities
    """
    device_info = {
        "device": str(device),
        "type": device.type,
        "index": device.index,
    }

    if device.type == "cuda":
        device_info.update({
            "name": torch.cuda.get_device_name(device),
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_reserved": torch.cuda.memory_reserved(device),
        })

    return device_info
