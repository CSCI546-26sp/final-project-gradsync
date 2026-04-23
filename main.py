import argparse
import time
import torch
import torch.nn as nn
import json

from pipeline import DistributedPipeline

import asyncio

import random
import numpy as np

def set_deterministic_seed(seed=42):
    """Locks all random number generators across all hardware backends."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Lock Apple Silicon (Mac Head Node)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    # Lock CUDA (if you ever switch Windows back to GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # Force deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. Define a standard, unmodified PyTorch Model
class MultiLayerTrans(nn.Module):
    def __init__(self, num_layers=4, d_model=1024, nhead=8, dim_feedforward=2048):
        super().__init__()
        # Using a ModuleList so the pipeline can easily slice it
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.0
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


def main():
    parser = argparse.ArgumentParser(description="Auto-Electing Distributed ML Pipeline")
    parser.add_argument('--host_address', type=str, required=True, help="This machine's IP:PORT (e.g., 192.168.1.50:12345)")
    parser.add_argument('--config', type=str, default='cluster.json', help="Path to cluster config file")
    args = parser.parse_args()

    # 1. Load the Configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {args.config}. Please create it!")
        return

    all_nodes = config.get("cluster_nodes", [])
    if args.host_address not in all_nodes:
        print(f"Warning: {args.host_address} is not listed in {args.config}!")
    
    peer_addresses = [addr for addr in all_nodes if addr != args.host_address]
    n_micro = config.get("n_micro", 4)

    print(f"--- Booting Node at {args.host_address} ---")
    print(f"Peers: {peer_addresses} | Micro-batches: {n_micro}")

    set_deterministic_seed(257)
    model = MultiLayerTrans(num_layers=4)

    # 2. Initialize Pipeline (Blocks until Raft Election is complete)
    print(f"Initializing pipeline and electing roles. Waiting on peers...")
    pipeline = DistributedPipeline(
        model=model,
        host_address=args.host_address,
        peer_addresses=peer_addresses,
        n_micro=n_micro
    )

    # 4. Diverged Execution based on role
    if pipeline.role == 'tail':
        # The Tail node gets trapped here, spinning up the gRPC server to listen for tensors
        print(f"Initialization complete. Serving pipeline slice on port {pipeline.local_port}...")
        pipeline.serve_forever()

    elif pipeline.role == 'head':
        # The Head node drives the actual training loop
        # print(f"Connecting to Tail node at {args.target_ip}:{args.port}...")
        
        async def train_loop():
            await pipeline._configure_remote()
            
            # Generate dummy data (Batch Size: 8, Seq Len: 32, Dim: 1024)
            print("Generating dummy dataset...")
            dummy_inputs = torch.randn(16, 8, 32, 1024)
            # Create a weak mathematical correlation so the loss actually decreases
            # dummy_targets = dummy_inputs[:, :, :, 0].mean(dim=-1, keepdim=True) + torch.randn(10, 8, 32, 1) * 0.1
            # Use 0:1 to keep the last dimension intact!
            dummy_targets = dummy_inputs[:, :, :, 0:1] * 0.5 + torch.randn(16, 8, 32, 1) * 0.1

            epochs = 3

            for epoch in range(epochs):
                print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
                epoch_loss = 0.0
                
                start_time = time.time()
                
                for batch_idx in range(len(dummy_inputs)):
                    x = dummy_inputs[batch_idx]
                    y = dummy_targets[batch_idx]

                    micro_x = torch.chunk(x, chunks=n_micro, dim=0)
                    micro_y = torch.chunk(y, chunks=n_micro, dim=0)

                    pipeline.zero_grad()

                    tasks = [pipeline.train_step(mx, my) for mx, my in zip(micro_x, micro_y)]
                    micro_losses = await asyncio.gather(*tasks)

                    pipeline.step()

                    batch_loss = sum(micro_losses) / len(micro_losses)
                    epoch_loss += batch_loss
                    
                    print(f"  Batch {batch_idx + 1}/16 | Loss: {batch_loss:.4f}")
                
                end_time = time.time()
                avg_loss = epoch_loss / len(dummy_inputs)
                print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.4f} | Time: {end_time - start_time:.2f}s")

        asyncio.run(train_loop())

if __name__ == '__main__':
    main()