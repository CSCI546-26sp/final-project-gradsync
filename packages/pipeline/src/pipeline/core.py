import torch
import torch.nn as nn
from .runner import HeadNodeRunner, TailNodeRunner
from .utils import detect_device
from orchestrator.node import ClusterNode


class DistributedPipeline(nn.Module):
    """
    The drop-in wrapper for Cross-OS distributed training.
    """

    def __init__(self, model: nn.Module, host_ip: str, peer_ips: list, port: int = 50051):
        super().__init__()
        self.host_ip = host_ip
        self.peer_ips = peer_ips
        self.port = port
        self.role = None
        self.runner = None
        self.device = detect_device()

        # Split the model up front; runner is initialized after join_cluster()
        self.layout = self._profile_and_split(model)

    def _profile_and_split(self, model):
        """Internal logic to slice the user's model."""
        # MVP split logic matching your original implementation
        layers = list(model.layers)
        split_idx = max(1, len(layers) // 2)

        head_slice = layers[:split_idx]
        tail_slice = layers[split_idx:]

        # --- NEW: Append the final output projection to the Tail node ---
        if hasattr(model, 'output_layer'):
            tail_slice.append(model.output_layer)

        return [head_slice, tail_slice]

    def _configure_remote(self):
        """Head node tells the tail node which layers it owns."""
        is_ready = self.runner.configure_remote(
            start_layer=len(self.layout[0]),
            end_layer=len(self.layout[0]) + len(self.layout[1])
        )
        if not is_ready:
            print("Warning: Remote Tail Node configuration failed or timed out.")

    def join_cluster(self):
        """Runs Raft coordinator election. Blocking until all nodes have agreed on roles."""
        node = ClusterNode(host_ip=self.host_ip, peer_ips=self.peer_ips, port=self.port)
        topology = node.join_cluster()

        # Determine role by comparing this node's IP against the elected coordinator
        if self.host_ip == topology.coordinator_ip:
            self.role = 'head'
            ordered = list(topology.ordered_node_ips)
            target_ip = ordered[1]  # next node in the pipeline
            self.runner = HeadNodeRunner(self.layout[0], target_ip, self.port, self.device)
            self._configure_remote()
        else:
            self.role = 'tail'
            self.runner = TailNodeRunner(self.layout[1], self.device)

        print(f"[{self.host_ip}] join_cluster complete. Role: {self.role.upper()}")

    def serve_forever(self):
        """Called by the Tail node to start listening for network tensors."""
        if self.role != 'tail':
            raise RuntimeError("Only the 'tail' node can serve.")
        self.runner.serve()

    def train_step(self, inputs, targets):
        """Called by the Head node to execute a distributed forward/backward pass."""
        if self.role != 'head':
            raise RuntimeError(
                "Only the 'head' node can initiate a train_step.")
        return self.runner.train_batch(inputs, targets)

    def parameters(self, recurse: bool = True):
        """Expose local parameters to the user's optimizer."""
        return self.runner.model_slice.parameters(recurse)
