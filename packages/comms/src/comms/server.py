import grpc
from concurrent import futures

# Assuming your generated protos are in a 'proto' folder inside 'comms'
from .proto import tensor_service_pb2
from .proto import tensor_service_pb2_grpc


class PipelineServer(tensor_service_pb2_grpc.PipelineRouterServicer):
    def __init__(self, processing_callback):
        # The callback accepts raw bytes/shapes and returns gradient bytes/shapes + loss
        self.processing_callback = processing_callback
        self.is_ready = False
        self.batch_counter = 0

    def AssignConfiguration(self, request, context):
        self.is_ready = True
        print(
            f"Assigned Layers: {request.start_layer_idx} to {request.end_layer_idx}")
        return tensor_service_pb2.ConfigAck(is_ready=True)

    def ExecutePipelineStage(self, request, context):
        self.batch_counter += 1

        # 1. Extract raw primitives
        act_shape = list(request.activation_shape)
        act_bytes = request.activation_bytes

        tgt_shape = list(request.target_shape)
        tgt_bytes = request.target_bytes

        # 2. Hand off pure bytes to the ML application logic in the pipeline package
        try:
            grad_bytes, grad_shape, loss_val = self.processing_callback(
                act_bytes, act_shape, tgt_bytes, tgt_shape
            )
        except Exception as e:
            # If the PyTorch callback crashes (e.g., OOM error), tell the client
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"ML Processing failed on server: {str(e)}")
            return tensor_service_pb2.BackwardPayload()

        # 3. Package the returned bytes back into a gRPC response
        return tensor_service_pb2.BackwardPayload(
            gradient_shape=grad_shape,
            gradient_bytes=grad_bytes,
            loss_value=loss_val
        )


def serve_pipeline(processing_callback, port=12345):
    """Starts the blocking gRPC server."""
    options = [
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ('grpc.max_send_message_length', 100 * 1024 * 1024)
    ]

    server = grpc.server(futures.ThreadPoolExecutor(
        max_workers=4), options=options)
    tensor_service_pb2_grpc.add_PipelineRouterServicer_to_server(
        PipelineServer(processing_callback), server
    )

    server.add_insecure_port(f'[::]:{port}')
    print(f"Pipeline Server listening on port {port}...")
    server.start()
    server.wait_for_termination()
