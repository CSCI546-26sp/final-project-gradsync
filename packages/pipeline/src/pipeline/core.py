import sys
import torch
import torch.nn as nn
from .runner import HeadNodeRunner, TailNodeRunner, MiddleNodeRunner
from .utils import detect_device
from orchestrator.node import ClusterNode
import time
import json
import asyncio

class DistributedPipeline(nn.Module):
    def __init__(self, model_builder, criterion: nn.Module, optim_class, optim_kwargs: dict, host_ip: str, elec_port: str, train_port: str, config_path: str):
        super().__init__()

        self.criterion = criterion
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find {config_path}. Please create it!")
            sys.exit(1)
        
        self.host_address = f"{host_ip}:{train_port}"
        self.election_host_address = f"{host_ip}:{elec_port}"
        
        raw_election = config.get("election_nodes", [])
        if self.election_host_address not in raw_election:
            print(f"Warning: {self.election_host_address} is not listed in {config_path}!")
        print(f"Election Nodes: {raw_election}")
        
        self.election_addresses = [addr for addr in raw_election if addr != self.election_host_address]

        raw_cluster = config.get("cluster_nodes", [])
        if self.host_address not in raw_cluster:
            print(f"Warning: {self.host_address} is not listed in {config_path}!")
        print(f"Cluster Nodes: {raw_cluster}")
        
        self.peer_addresses = [addr for addr in raw_cluster if addr != self.host_address]
        self.n_micro = config.get("n_micro", 4)
    
        print(f"Peers: {self.peer_addresses} | Micro-batches: {self.n_micro}")
        
        self.local_ip, self.local_port_str = self.host_address.split(':')
        self.local_port = int(self.local_port_str)
        
        self.device = detect_device()
        self.role = None
        self.runner = None

        layer_memory_reqs = self._meta_profile(model_builder)

        self.join_cluster(model_builder, layer_memory_reqs)

        self.runner.optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)

        self.start()

    def _meta_profile(self, model_builder):
        with torch.device('meta'):
            meta_model = model_builder()
        
        layer_memory_reqs = []
        for layer in meta_model.layers:
            layer_params = sum(p.numel() * p.element_size() for p in layer.parameters())
            # x4 rough estimate for gradients and optimizer states
            layer_memory_reqs.append(layer_params * 4)
            
        return layer_memory_reqs

    async def _configure_remote(self):
        is_ready = await self.runner.configure_remote(
            start_layer=len(self.layout[0]),
            end_layer=len(self.layout[0]) + len(self.layout[1])
        )
        if not is_ready:
            print("Warning: Remote Tail Node configuration failed or timed out.")

    def join_cluster(self, model_builder, layer_memory_reqs):
        print(f"[{self.host_address}] Initiating Cluster Election...")
        node = ClusterNode(host_ip=self.election_host_address, peer_ips=self.election_addresses)
        topology, capacities = node.join_cluster()

        idx = topology.node_index
        total_nodes = len(topology.ordered_node_ips)
        total_layers = len(layer_memory_reqs)

        if idx == 0:
            self.role = 'head'
            # Leader computes partition boundaries
            allocations = {}
            current_layer = 0
            for ip in topology.ordered_node_ips:
                start_idx = current_layer
                cap = capacities.get(ip, float('inf'))
                
                while current_layer < total_layers and cap >= layer_memory_reqs[current_layer]:
                    cap -= layer_memory_reqs[current_layer]
                    current_layer += 1
                    
                if start_idx == current_layer and current_layer < total_layers:
                    current_layer += 1
                    
                allocations[ip] = {'start': start_idx, 'end': current_layer}
            
            if current_layer < total_layers:
                print(f"WARNING: Cluster Out of Memory! Could only fit {current_layer}/{total_layers} layers.")

            node.broadcast_partitioning(allocations)
            partition_config = node.partition_config
        else:
            if idx == total_nodes - 1:
                self.role = 'tail'
            else:
                self.role = 'middle'
            
            print(f"[{self.host_address}] Waiting for leader to partition layers...")
            partition_config = node.wait_for_partitioning()

        start_layer = partition_config.start_layer_idx
        end_layer = partition_config.end_layer_idx

        real_model = model_builder()
        layers = list(real_model.layers)
        my_slice = layers[start_layer:end_layer]

        # Parse next node details
        next_ip, next_port = None, None
        if topology.next_node_idx >= 0 and topology.next_node_idx < len(self.peer_addresses):
            next_node_address = self.peer_addresses[topology.next_node_idx]
            next_ip, next_port_str = next_node_address.split(':')
            next_port = int(next_port_str)

        if self.role == 'head':
            self.runner = HeadNodeRunner(my_slice, target_ip=next_ip, port=next_port, device=self.device)
        elif self.role == 'tail':
            if hasattr(real_model, 'output_layer'):
                my_slice.append(real_model.output_layer)
            self.runner = TailNodeRunner(my_slice, device=self.device, criterion=self.criterion, n_micro=self.n_micro)
            self.serve_port = self.local_port 
        else:
            self.runner = MiddleNodeRunner(my_slice, target_ip=next_ip, port=next_port, device=self.device, n_micro=self.n_micro)
            self.serve_port = self.local_port

        print(f"[{self.host_address}] Assigned Role: {self.role.upper()} | Layers: {start_layer} to {end_layer}")

    def serve_forever(self, port):
        return self.runner.serve(port=port)

    async def train_step(self, inputs, targets):
        if self.role != 'head':
            raise RuntimeError("Only the 'head' node can initiate a train_step.")
        return await self.runner.train_batch(inputs, targets)

    def parameters(self, recurse: bool = True):
        return self.runner.model_slice.parameters(recurse)

    def zero_grad(self):
        if self.role == 'head':
            self.runner.optimizer.zero_grad()

    def step(self):
        if self.role == 'head':
            self.runner.optimizer.step()

    def start(self):
        if self.role in ['middle', 'tail']:
            print(f"[{self.host_address}] Worker Node activated. Serving on port {self.local_port}...")
            
            try:
                self.serve_forever(port=self.local_port)
                
            except KeyboardInterrupt:
                print(f"\n[{self.host_address}] Shutting down worker node cleanly...")
                sys.exit(0)
            
        elif self.role == 'head':
            print(f"[{self.host_address}] Head Node activated. Configuring downstream connections...")
            asyncio.run(self._configure_remote())
            print(f"[{self.host_address}] Cluster linked! Ready for training.")

    def execute_batch(self, inputs, targets):
        if self.role != 'head':
            raise RuntimeError("Only the head node can execute training batches.")
        return asyncio.run(self._async_execute_batch(inputs, targets))

    async def _async_execute_batch(self, inputs, targets):
        micro_x = torch.chunk(inputs, chunks=self.n_micro, dim=0)
        micro_y = torch.chunk(targets, chunks=self.n_micro, dim=0)

        self.zero_grad()

        tasks = [self.train_step(mx, my) for mx, my in zip(micro_x, micro_y)]
        micro_losses = await asyncio.gather(*tasks)

        self.step()

        return sum(micro_losses) / len(micro_losses)