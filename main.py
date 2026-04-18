import argparse
import time
import torch
import torch.nn as nn

from pipeline import DistributedPipeline

import asyncio

# 1. Define a standard, unmodified PyTorch Model
class MultiLayerTrans(nn.Module):
    def __init__(self, num_layers=4, d_model=1024, nhead=8, dim_feedforward=2048):
        super().__init__()
        # Using a ModuleList so the pipeline can easily slice it
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


def main():
    parser = argparse.ArgumentParser(description="Test the Distributed ML Pipeline Data Path")
    parser.add_argument('--role', type=str, required=True, choices=['head', 'middle', 'tail'], help="Role of this node")
    parser.add_argument('--target_ip', type=str, default='127.0.0.1', help="IP of the Tail node (used by Head)")
    parser.add_argument('--port', type=int, default=12345, help="Port for gRPC communication")
    args = parser.parse_args()

    print(f"--- Booting as {args.role.upper()} NODE ---")

    # 2. Instantiate the model
    model = MultiLayerTrans(num_layers=4)

    # 3. Wrap it in your library's pipeline
    pipeline = DistributedPipeline(
        model=model,
        role=args.role,
        target_ip=args.target_ip,
        port=args.port
    )

    # 4. Diverged Execution based on role
    if args.role in ['tail', 'middle']:
        # The Tail node gets trapped here, spinning up the gRPC server to listen for tensors
        print(f"Initialization complete. Serving pipeline slice on port {args.port}...")
        pipeline.serve_forever()

    elif args.role == 'head':
        # The Head node drives the actual training loop
        print(f"Connecting to Tail node at {args.target_ip}:{args.port}...")
        
        async def train_loop():
            # Generate dummy data (Batch Size: 8, Seq Len: 32, Dim: 1024)
            print("Generating dummy dataset...")
            dummy_inputs = torch.randn(10, 8, 32, 1024)
            # Create a weak mathematical correlation so the loss actually decreases
            # dummy_targets = dummy_inputs[:, :, :, 0].mean(dim=-1, keepdim=True) + torch.randn(10, 8, 32, 1) * 0.1
            # Use 0:1 to keep the last dimension intact!
            dummy_targets = dummy_inputs[:, :, :, 0:1] * 0.5 + torch.randn(10, 8, 32, 1) * 0.1

            epochs = 3

            num_micro_batches = 4

            for epoch in range(epochs):
                print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
                epoch_loss = 0.0
                
                start_time = time.time()
                
                for batch_idx in range(len(dummy_inputs)):
                    x = dummy_inputs[batch_idx]
                    y = dummy_targets[batch_idx]

                    micro_x = torch.chunk(x, chunks=num_micro_batches, dim=0)
                    micro_y = torch.chunk(y, chunks=num_micro_batches, dim=0)

                    tasks = [pipeline.train_step(mx, my) for mx, my in zip(micro_x, micro_y)]

                    micro_losses = await asyncio.gather(*tasks)

                    batch_loss = sum(micro_losses) / len(micro_losses)
                    
                    
                    epoch_loss += batch_loss
                    
                    print(f"  Batch {batch_idx + 1}/10 | Loss: {batch_loss:.4f}")
                
                end_time = time.time()
                avg_loss = epoch_loss / len(dummy_inputs)
                print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.4f} | Time: {end_time - start_time:.2f}s")

        asyncio.run(train_loop())

if __name__ == '__main__':
    main()