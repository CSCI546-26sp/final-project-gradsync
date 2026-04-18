import torch
import torch.nn as nn
from .runner import HeadNodeRunner, TailNodeRunner, MiddleNodeRunner
from .utils import detect_device

import asyncio


class DistributedPipeline(nn.Module):
    """
    The drop-in wrapper for Cross-OS distributed training.
    """

    def __init__(self, model: nn.Module, role: str, target_ip: str = "127.0.0.1", port: int = 12345):
        super().__init__()
        self.role = role.lower()

        # 1. Profile and split the user's model
        self.layout = self._profile_and_split(model)

        # 2. Initialize the correct execution engine
        self.device = detect_device()
        if self.role == 'tail':
            self.runner = TailNodeRunner(self.layout[2], self.device)

        elif self.role == 'head':
            self.runner = HeadNodeRunner(
                self.layout[0], target_ip, port, self.device)
            asyncio.run(self._configure_remote())
        elif self.role == 'middle':
            self.runner = MiddleNodeRunner(self.layout[1], target_ip, port, self.device)
        else:
            raise ValueError("Role must be 'head' or 'tail'.")

    def _profile_and_split(self, model):
        """Internal logic to slice the user's model."""
        # MVP split logic matching your original implementation
        layers = list(model.layers)
        split_idx = max(1, len(layers) // 2)

        head_slice = layers[:split_idx]
        ### middle slice
        # middle_slice = layers[split_idx:split_idx+1]

        tail_slice = layers[split_idx:]

        # --- NEW: Append the final output projection to the Tail node ---
        if hasattr(model, 'output_layer'):
            tail_slice.append(model.output_layer)

        return [head_slice, middle_slice, tail_slice]

    async def _configure_remote(self):
        """Head node tells the tail node which layers it owns."""
        is_ready = self.runner.configure_remote(
            start_layer=len(self.layout[0]),
            end_layer=len(self.layout[0]) + len(self.layout[1])
        )
        if not is_ready:
            print("Warning: Remote Tail Node configuration failed or timed out.")

    def serve_forever(self):
        """Called by the Tail node to start listening for network tensors."""
        # if self.role != 'tail':
        #     raise RuntimeError("Only the 'tail' node can serve.")
        self.runner.serve()

    async def train_step(self, inputs, targets):
        """Called by the Head node to execute a distributed forward/backward pass."""
        if self.role != 'head':
            raise RuntimeError(
                "Only the 'head' node can initiate a train_step.")
        return await self.runner.train_batch(inputs, targets)

    def parameters(self, recurse: bool = True):
        """Expose local parameters to the user's optimizer."""
        return self.runner.model_slice.parameters(recurse)
