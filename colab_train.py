import argparse
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pipeline import DistributedPipeline

class HFBlockAdapter(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        
    def forward(self, x):
        return self.block(x)[0]

class DistributedSmolLM(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        
        layers = [hf_model.model.embed_tokens]
        for block in hf_model.model.layers:
            layers.append(HFBlockAdapter(block))
        layers.append(hf_model.model.norm)
        
        self.layers = nn.ModuleList(layers)
        self.output_layer = hf_model.lm_head

class DistributedCausalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        labels = labels.long()
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host_ip', type=str, required=True)
    parser.add_argument("--elec_port", type=str, required=True)
    parser.add_argument("--train_port", type=str, required=True)
    parser.add_argument('--config', type=str, default='cluster.json')
    args = parser.parse_args()

    print("Loading HuggingFace model...")
    model_name = "HuggingFaceTB/SmolLM-360M"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("Initializing pipeline and electing roles. Waiting on peers...")
    model_builder = lambda: DistributedSmolLM(hf_model)
    criterion = DistributedCausalLoss()

    pipeline = DistributedPipeline(
        model_builder=model_builder,
        criterion=criterion,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.001},
        host_ip=args.host_ip,
        elec_port=args.elec_port,
        train_port=args.train_port,
        config_path=args.config
    )

    print("Setting up tokenizer and dataset...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Loading dataset...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [text for text in dataset['text'] if len(text) > 50]

    batch_size = 4
    seq_len = 512
    epochs = 3
    batches_per_epoch = 5 

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        for b in range(batches_per_epoch):
            start_time = time.time()
            
            batch_texts = texts[b * batch_size : (b + 1) * batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", max_length=seq_len, padding="max_length", truncation=True)
            
            input_ids = inputs.input_ids
            labels = input_ids.clone().float()

            loss = pipeline.execute_batch(input_ids, labels)
            
            end_time = time.time()
            
            if torch.cuda.is_available():
                peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"  Batch {b+1}/{batches_per_epoch} | Loss: {loss:.4f} | Time: {end_time - start_time:.2f}s | VRAM: {peak_vram_gb:.2f} GB")
            else:
                print(f"  Batch {b+1}/{batches_per_epoch} | Loss: {loss:.4f} | Time: {end_time - start_time:.2f}s")

if __name__ == '__main__':
    main()