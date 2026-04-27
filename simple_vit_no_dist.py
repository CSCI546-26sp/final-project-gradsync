import argparse
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from pipeline import DistributedPipeline

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

class PureTransformer(nn.Module):
    def _init_(self, num_layers=12, d_model=768, nhead=12):
        super()._init_()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(28, d_model))
        
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
                    batch_first=True, activation="gelu"
                )
            )
            
        self.layers.append(nn.Flatten(start_dim=1))
        
        self.layers.append(nn.Linear(28 * d_model, 10))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    print("Downloading/Loading MNIST Dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    full_images, full_labels = next(iter(train_loader))

    model = PureTransformer(num_layers=4, d_model=768)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    batch_size = 64

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        epoch_loss = 0.0
        start_time = time.time()
        
        indices = torch.randperm(len(full_images))
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) < batch_size: continue
            
            x = full_images[batch_idx].squeeze(1)
            y = full_labels[batch_idx]

            out = model(x)
            loss = criterion(out, y)
            epoch_loss += loss

            if (i // batch_size) % 20 == 0:
                print(f"  Batch {(i // batch_size) + 1}/{(len(full_images)//batch_size)} | Loss: {loss:.4f}")
        
        end_time = time.time()
        avg_loss = epoch_loss / (len(full_images) // batch_size)
        print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.4f} | Time: {end_time - start_time:.2f}s")

if __name__ == '_main_':
    main()