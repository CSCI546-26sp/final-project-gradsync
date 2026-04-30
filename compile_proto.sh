#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

case "$(uname -s 2>/dev/null || echo unknown)" in
  Darwin*) platform="macOS" ;;
  Linux*) platform="Linux" ;;
  MINGW*|MSYS*|CYGWIN*) platform="Windows POSIX shell" ;;
  *) platform="unknown POSIX shell" ;;
esac

echo "Generating protobuf stubs on ${platform}..."

mkdir -p packages/comms/src/comms/proto
uv run python -m grpc_tools.protoc \
  -I packages/comms/proto \
  --python_out=packages/comms/src/comms/proto \
  --grpc_python_out=packages/comms/src/comms/proto \
  packages/comms/proto/tensor_service.proto

mkdir -p packages/orchestrator/src/orchestrator/proto
uv run python -m grpc_tools.protoc \
  -I packages/orchestrator/proto \
  --python_out=packages/orchestrator/src/orchestrator/proto \
  --grpc_python_out=packages/orchestrator/src/orchestrator/proto \
  packages/orchestrator/proto/cluster_service.proto

uv run python - <<'PY'
from pathlib import Path

patches = {
    Path("packages/comms/src/comms/proto/tensor_service_pb2_grpc.py"): (
        "import tensor_service_pb2 as",
        "from . import tensor_service_pb2 as",
    ),
    Path("packages/orchestrator/src/orchestrator/proto/cluster_service_pb2_grpc.py"): (
        "import cluster_service_pb2 as",
        "from . import cluster_service_pb2 as",
    ),
}

for path, (old, new) in patches.items():
    text = path.read_text()
    path.write_text(text.replace(old, new))
PY

echo "Generated protobuf stubs successfully."
