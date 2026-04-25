import torch
import torch.nn as nn
from comms.server import serve_pipeline
from comms.client import PipelineClient
from .utils import pack_tensor, unpack_tensor
import asyncio

class BuddyNode(nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.local_layers = nn.ModuleList(layer_list)

    def forward(self, x):
        for layer in self.local_layers:
            x = layer(x)
        return x

class TailNodeRunner:
    def __init__(self, model_slice_layers, device, criterion, n_micro=1):
        self.device = device
        self.model_slice = BuddyNode(model_slice_layers).to(self.device)
        self.criterion = criterion
        self.optimizer = None
        self.accum_steps = n_micro
        self.fw_started = 0
        self.bw_completed = 0

    async def _process_batch_callback(self, act_bytes, act_shape, tgt_bytes, tgt_shape):
        if self.fw_started % self.accum_steps == 0:
            self.optimizer.zero_grad()
        self.fw_started += 1

        activations = unpack_tensor(act_bytes, act_shape, self.device)
        activations.requires_grad_(True)
        
        targets = unpack_tensor(tgt_bytes, tgt_shape, self.device)

        outputs = self.model_slice(activations)
        loss = self.criterion(outputs, targets)

        scaled_loss = loss / self.accum_steps
        scaled_loss.backward()

        self.bw_completed += 1
        if self.bw_completed % self.accum_steps == 0:
            self.optimizer.step()

        grad_bytes, grad_shape = pack_tensor(activations.grad)
        return grad_bytes, grad_shape, loss.item()

    def serve(self, port=12345):
        print(f"Tail Node Engine ready. Listening on port {port} (Device: {self.device})...")
        return serve_pipeline(processing_callback=self._process_batch_callback, port=port)

class HeadNodeRunner:
    def __init__(self, model_slice_layers, target_ip, port=12345, device=None):
        self.device = device or torch.device("cpu")
        self.model_slice = BuddyNode(model_slice_layers).to(self.device)
        self.optimizer = None
        self.client = PipelineClient(target_ip=target_ip, port=port)

    async def configure_remote(self, start_layer, end_layer):
        return await self.client.send_pipeline_config(start_layer, end_layer, is_tail=True)

    async def train_batch(self, inputs, targets):
        inputs = inputs.to(self.device)
        
        local_activations = self.model_slice(inputs)
        
        act_bytes, act_shape = pack_tensor(local_activations)
        tgt_bytes, tgt_shape = pack_tensor(targets)

        grad_bytes, grad_shape, loss_val = await self.client.send_forward_receive_backward(
            act_bytes, act_shape, tgt_bytes, tgt_shape
        )

        returned_grads = unpack_tensor(grad_bytes, grad_shape, self.device)

        local_activations.backward(returned_grads)

        return loss_val
    
class MiddleNodeRunner:
    def __init__(self, model_slice_layers, target_ip, port, device, n_micro=1):
        self.device = device
        self.model_slice = BuddyNode(model_slice_layers).to(self.device)
        self.optimizer = None
        self.client = PipelineClient(target_ip=target_ip, port=port)

        self.accum_steps = n_micro
        self.fw_started = 0
        self.bw_completed = 0
        self.mb_counter = 0

    async def _process_batch_callback(self, act_bytes, act_shape, tgt_bytes, tgt_shape):
        self.mb_counter += 1
        mb_id = self.mb_counter

        if self.fw_started % self.accum_steps == 0:
            self.optimizer.zero_grad()
        self.fw_started += 1

        print(f"  [MB {mb_id}] FORWARD Started")

        activations = unpack_tensor(act_bytes, act_shape, self.device)
        activations.requires_grad_(True)
        
        local_output = self.model_slice(activations)

        print(f"  [MB {mb_id}] FORWARD Done. Yielding to network (waiting on Tail...)")
        next_act_bytes, next_act_shape = pack_tensor(local_output)
        grad_bytes, grad_shape, loss_val = await self.client.send_forward_receive_backward(
            next_act_bytes, next_act_shape, tgt_bytes, tgt_shape
        )
        
        print(f"  [MB {mb_id}] BACKWARD Received from Tail. Resuming...")
        remote_grads = unpack_tensor(grad_bytes, grad_shape, self.device)
        
        local_output.backward(remote_grads)

        self.bw_completed += 1
        if self.bw_completed % self.accum_steps == 0:
            self.optimizer.step()
        
        print(f"  [MB {mb_id}] BACKWARD Done. Returning gradients to Head.")
        my_grad_bytes, my_grad_shape = pack_tensor(activations.grad)

        return my_grad_bytes, my_grad_shape, loss_val

    def serve(self, port=12345):
        return serve_pipeline(processing_callback=self._process_batch_callback, port=port)