import torch
import torch.nn as nn
from .runner import HeadNodeRunner, TailNodeRunner, MiddleNodeRunner
from .utils import detect_device
from orchestrator.node import ClusterNode
import time

import asyncio


class DistributedPipeline(nn.Module):
    def __init__(self, model: nn.Module, host_address: str, election_host_address: str, election_addresses: list, peer_addresses: list, n_micro: int = 4):
        super().__init__()
        self.host_address = host_address
        self.election_host_address = election_host_address
        self.election_addresses = election_addresses
        self.peer_addresses = peer_addresses
        self.n_micro = n_micro
        
        # Split local address
        self.local_ip, self.local_port_str = self.host_address.split(':')
        self.local_port = int(self.local_port_str)
        
        self.device = detect_device()
        self.layout = self._profile_and_split(model)
        
        self.role = None
        self.runner = None

        # Execute Raft Election immediately
        self.join_cluster()

    def _profile_and_split(self, model):
        """Internal logic to slice the user's model."""
        # MVP split logic matching your original implementation
        layers = list(model.layers)
        split_idx = max(1, len(layers) // 2)

        head_slice = layers[:split_idx]
        ### middle slice
        middle_slice = layers[split_idx:split_idx+1]

        tail_slice = layers[split_idx+1:]

        # --- NEW: Append the final output projection to the Tail node ---
        if hasattr(model, 'output_layer'):
            tail_slice.append(model.output_layer)

        return [head_slice, middle_slice, tail_slice]
        # return [head_slice, tail_slice]

    async def _configure_remote(self):
        """Head node tells the tail node which layers it owns."""
        is_ready = self.runner.configure_remote(
            start_layer=len(self.layout[0]),
            end_layer=len(self.layout[0]) + len(self.layout[1])
        )
        if not is_ready:
            print("Warning: Remote Tail Node configuration failed or timed out.")

    def join_cluster(self):
        print(f"[{self.host_address}] Initiating Cluster Election...")
        node = ClusterNode(host_ip=self.election_host_address, peer_ips=self.election_addresses)
        topology = node.join_cluster()

        node.shutdown()  # We only needed the election result, so we can shut down the Raft node now
        # Parse next node details
        next_ip, next_port = None, None
        if topology.next_node_idx >= 0 and topology.next_node_idx < len(self.peer_addresses):
            # next_node_address = topology.ordered_node_ips[topology.next_node_idx]
            next_node_address = self.peer_addresses[topology.next_node_idx]
            next_ip, next_port_str = next_node_address.split(':')
            next_port = int(next_port_str)

        idx = topology.node_index
        total_nodes = len(topology.ordered_node_ips)

        if idx == 0:
            self.role = 'head'
            self.runner = HeadNodeRunner(self.layout[0], target_ip=next_ip, port=next_port, device=self.device)
        elif idx == total_nodes - 1:
            self.role = 'tail'
            self.runner = TailNodeRunner(self.layout[2], device=self.device, n_micro=self.n_micro)
            self.serve_port = self.local_port 
        else:
            self.role = 'middle'
            self.runner = MiddleNodeRunner(self.layout[1], target_ip=next_ip, port=next_port, device=self.device, n_micro=self.n_micro)
            self.serve_port = self.local_port

        print(f"[{self.host_address}] Election complete! Assigned Role: {self.role.upper()}")

    def serve_forever(self, port):
        """Called by the Tail node to start listening for network tensors."""
        # if self.role != 'tail':
        #     raise RuntimeError("Only the 'tail' node can serve.")
        self.runner.serve(port=port)

    async def train_step(self, inputs, targets):
        """Called by the Head node to execute a distributed forward/backward pass."""
        if self.role != 'head':
            raise RuntimeError(
                "Only the 'head' node can initiate a train_step.")
        return await self.runner.train_batch(inputs, targets)

    def parameters(self, recurse: bool = True):
        """Expose local parameters to the user's optimizer."""
        return self.runner.model_slice.parameters(recurse)

    def zero_grad(self):
        if self.role == 'head':
            self.runner.optimizer.zero_grad()

    def step(self):
        if self.role == 'head':
            self.runner.optimizer.step()