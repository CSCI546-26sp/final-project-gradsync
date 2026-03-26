mkdir -p packages/comms/src/comms/proto && \
uv run python -m grpc_tools.protoc \
  -I packages/comms/proto \
  --python_out=packages/comms/src/comms/proto \
  --grpc_python_out=packages/comms/src/comms/proto \
  packages/comms/proto/tensor_service.proto