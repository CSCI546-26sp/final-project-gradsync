import grpc
from .proto import tensor_service_pb2
from .proto import tensor_service_pb2_grpc


class PipelineClient:
    def __init__(self, target_ip="localhost", port=12345):
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024)
        ]

        self.target = f"{target_ip}:{port}"
        self.options = options

        self.channel = None
        self.stub = None

    def _connect(self):
        # Establish the connection to the next node in the sequence
        self.channel = grpc.aio.insecure_channel(
            self.target, options=self.options)
        self.stub = tensor_service_pb2_grpc.PipelineRouterStub(self.channel)



    async def send_pipeline_config(self, start_layer, end_layer, is_tail=True):
        self._connect()
        request = tensor_service_pb2.SplitConfig(
            start_layer_idx=start_layer,
            end_layer_idx=end_layer,
            is_tail_node=is_tail
        )
        try:
            response = await self.stub.AssignConfiguration(request)
            return response.is_ready
        except grpc.RpcError as e:
            print(
                f"Failed to configure remote node {self.channel._channel.target()}: {e.details()}")
            return False

    async def send_forward_receive_backward(self, act_bytes, act_shape, target_bytes, target_shape):
        """Blocks until the remote server finishes the forward/backward pass and returns gradients."""
        self._connect()
        request = tensor_service_pb2.ForwardPayload(
            activation_shape=act_shape,
            activation_bytes=act_bytes,
            target_shape=target_shape,
            target_bytes=target_bytes
        )

        request_tensor_bytes = len(act_bytes) + len(target_bytes)
        request_proto_bytes = len(request.SerializeToString())

        # print(f"request_tensor_bytes: {request_tensor_bytes}")
        print(f"request_proto_bytes: {request_proto_bytes}")

        try:
            response = await self.stub.ExecutePipelineStage(request)


            response_proto_bytes = len(response.SerializeToString())
            response_tensor_bytes = len(response.gradient_bytes)

            print(f"response_proto_bytes: {response_proto_bytes}")
            # print(f"response_tensor_bytes: {response_tensor_bytes}")

            return response.gradient_bytes, list(response.gradient_shape), response.loss_value
        except grpc.RpcError as e:
            print(f"Pipeline transmission failed: {e.details()}")
            raise e

    async def close(self):
        """Cleanly shut down the channel."""
        if self.channel is not None:
            await self.channel.close()
