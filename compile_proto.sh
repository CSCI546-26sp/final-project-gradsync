mkdir -p packages/comms/src/comms/proto && \
uv run python -m grpc_tools.protoc \
  -I packages/comms/proto \
  --python_out=packages/comms/src/comms/proto \
  --grpc_python_out=packages/comms/src/comms/proto \
  packages/comms/proto/tensor_service.proto
sed -i 's/^import \(.*_pb2\) as/from . import \1 as/' \
  packages/comms/src/comms/proto/tensor_service_pb2_grpc.py

mkdir -p packages/orchestrator/src/orchestrator/proto && \
uv run python -m grpc_tools.protoc \
  -I packages/orchestrator/proto \
  --python_out=packages/orchestrator/src/orchestrator/proto \
  --grpc_python_out=packages/orchestrator/src/orchestrator/proto \
  packages/orchestrator/proto/cluster_service.proto
sed -i 's/^import \(.*_pb2\) as/from . import \1 as/' \
  packages/orchestrator/src/orchestrator/proto/cluster_service_pb2_grpc.py