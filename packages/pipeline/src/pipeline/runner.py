import torch
import torch.nn as nn

# Import from our newly decoupled comms package
from comms.server import serve_pipeline
from comms.client import PipelineClient

from .utils import pack_tensor, unpack_tensor

import asyncio

class BuddyNode(nn.Module):
    """Wraps a specific slice of the user's model for local execution."""
    def __init__(self, layer_list):
        super().__init__()
        self.local_layers = nn.ModuleList(layer_list)

    def forward(self, x):
        for layer in self.local_layers:
            x = layer(x)
        return x

class TailNodeRunner:
    """Handles the backward pass and acts as the gRPC server."""
    def __init__(self, model_slice_layers, device, lr=0.01):
        self.device = device
        self.model_slice = BuddyNode(model_slice_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model_slice.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    async def _process_batch_callback(self, act_bytes, act_shape, tgt_bytes, tgt_shape):
        """Translates raw network bytes into PyTorch operations."""

        # 1. Deserialize Bytes -> Tensors (Wrap with bytearray!)
        # activations = torch.frombuffer(bytearray(act_bytes), dtype=torch.float32).reshape(act_shape).clone()
        activations = unpack_tensor(act_bytes, act_shape, self.device)
        activations = activations.to(self.device)
        activations.requires_grad_(True)
        
        # targets = torch.frombuffer(bytearray(tgt_bytes), dtype=torch.float32).reshape(tgt_shape).clone()
        # targets = targets.to(self.device)

        targets = unpack_tensor(tgt_bytes, tgt_shape, self.device)

        # 2. Local Training Loop
        self.optimizer.zero_grad()
        outputs = self.model_slice(activations)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        # 3. Serialize Gradients -> Bytes
        # grad_bytes = activations.grad.cpu().numpy().tobytes()
        # grad_shape = list(activations.grad.shape)

        grad_bytes, grad_shape = pack_tensor(activations.grad)

        return grad_bytes, grad_shape, loss.item()

    def serve(self, port=12345):
        print(f"Tail Node Engine ready. Listening on port {port} (Device: {self.device})...")
        # Start the dumb comms server and pass it our PyTorch callback
        serve_pipeline(processing_callback=self._process_batch_callback, port=port)

class HeadNodeRunner:
    """Handles the initial forward pass and acts as the gRPC client."""
    def __init__(self, model_slice_layers, target_ip, port=12345, device=None, lr=0.01):
        self.device = device or torch.device("cpu")
        self.model_slice = BuddyNode(model_slice_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model_slice.parameters(), lr=lr)
        
        # Initialize the dumb comms client
        self.client = PipelineClient(target_ip=target_ip, port=port)

    async def configure_remote(self, start_layer, end_layer):
        return self.client.send_pipeline_config(start_layer, end_layer, is_tail=True)

    async def train_batch(self, inputs, targets):
        """Executes one distributed forward/backward pass."""


        inputs = inputs.to(self.device)
        self.optimizer.zero_grad()
        
        # 1. Local Forward Pass
        local_activations = self.model_slice(inputs)
        
        # 2. Serialize Tensors -> Bytes
        # act_bytes = local_activations.cpu().detach().numpy().tobytes()
        # act_shape = list(local_activations.shape)
        
        # tgt_bytes = targets.cpu().detach().numpy().tobytes()
        # tgt_shape = list(targets.shape)

        act_bytes, act_shape = pack_tensor(local_activations)
        tgt_bytes, tgt_shape = pack_tensor(targets)

        # 3. Transmit and block for Tail node's response
        grad_bytes, grad_shape, loss_val = self.client.send_forward_receive_backward(
            act_bytes, act_shape, tgt_bytes, tgt_shape
        )

        # 4. Deserialize returned Bytes -> Tensors (Wrap with bytearray!)
        # returned_grads = torch.frombuffer(bytearray(grad_bytes), dtype=torch.float32).reshape(grad_shape).clone()
        # returned_grads = returned_grads.to(self.device)
        returned_grads = unpack_tensor(grad_bytes, grad_shape, self.device)

        # 5. Local Backward Pass
        local_activations.backward(returned_grads)
        self.optimizer.step()

        return loss_val
    
class MiddleNodeRunner:
    def __init__(self, model_slice_layers, target_ip, port, device, lr=0.01):
        self.device = device
        self.model_slice = BuddyNode(model_slice_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model_slice.parameters(), lr=lr)
        
        # Middle node is a hybrid: it needs a client to talk to the NEXT node
        self.client = PipelineClient(target_ip=target_ip, port=port)

        self.mb_counter = 0

    async def _process_batch_callback(self, act_bytes, act_shape, tgt_bytes, tgt_shape):
        self.mb_counter += 1
        mb_id = self.mb_counter

        print(f"  [MB {mb_id}] FORWARD Started")

        # 1. Unpack what we got from the PREVIOUS node
        activations = unpack_tensor(act_bytes, act_shape, self.device)
        activations.requires_grad_(True)
        
        # 2. Local Forward Pass
        local_output = self.model_slice(activations)

        await asyncio.sleep(0.5) #### REMOVE WHEN TESTING
        print("WAITING HERE CUS DID NOT COMMENT FORCE WAIT")
        print(f"  [MB {mb_id}] FORWARD Done. Yielding to network (waiting on Tail...)")
        # 3. Relay to the NEXT node (The "Ping")
        next_act_bytes, next_act_shape = pack_tensor(local_output)
        grad_bytes, grad_shape, loss_val = self.client.send_forward_receive_backward(
            next_act_bytes, next_act_shape, tgt_bytes, tgt_shape
        )
        
        print(f"  [MB {mb_id}] BACKWARD Received from Tail. Resuming...")
        # 4. Unpack returned gradients from the NEXT node
        remote_grads = unpack_tensor(grad_bytes, grad_shape, self.device)
        

        # 5. Local Backward Pass
        self.optimizer.zero_grad()
        local_output.backward(remote_grads)
        self.optimizer.step()
        await asyncio.sleep(0.5) #### REMOVE WHEN TESTING
        print("WAITING HERE CUS DID NOT COMMENT FORCE WAIT")
        print(f"  [MB {mb_id}] BACKWARD Done. Returning gradients to Head.")
        # 6. Send our gradients back to the PREVIOUS node (The "Pong")
        my_grad_bytes, my_grad_shape = pack_tensor(activations.grad)

        return my_grad_bytes, my_grad_shape, loss_val

    def serve(self, port=12345):
        # Starts listening for the node behind it
        serve_pipeline(processing_callback=self._process_batch_callback, port=port)