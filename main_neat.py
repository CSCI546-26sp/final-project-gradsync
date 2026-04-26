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
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiLayerTrans(nn.Module):
    def __init__(self, num_layers=4, d_model=1024, nhead=8, dim_feedforward=2048):
        super().__init__()
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--host_ip', type=str, required=True)
    parser.add_argument("--elec_port", type=str, required=True)
    parser.add_argument("--train_port", type=str, required=True)
    parser.add_argument('--config', type=str, default='cluster.json')
    args = parser.parse_args()

    set_deterministic_seed(257)
    model_builder = lambda: MultiLayerTrans(num_layers=4)
    criterion = nn.MSELoss()

    print(f"Initializing pipeline and electing roles. Waiting on peers...")
    pipeline = DistributedPipeline(
        model_builder=model_builder,
        criterion=criterion,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.01},
        host_ip=args.host_ip,
        elec_port=args.elec_port,
        train_port=args.train_port,
        config_path=args.config
    )

    print("Generating dummy dataset...")
    dummy_inputs = torch.randn(16, 8, 32, 1024)
    dummy_targets = dummy_inputs[:, :, :, 0:1] * 0.5 + torch.randn(16, 8, 32, 1) * 0.1

    epochs = 3

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx in range(len(dummy_inputs)):
            x = dummy_inputs[batch_idx]
            y = dummy_targets[batch_idx]

            loss = pipeline.execute_batch(x, y)

            epoch_loss += loss
            print(f"  Batch {batch_idx + 1}/16 | Loss: {loss:.4f}")
        
        end_time = time.time()
        avg_loss = epoch_loss / len(dummy_inputs)
        print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.4f} | Time: {end_time - start_time:.2f}s")

if __name__ == '__main__':
    main()