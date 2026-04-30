# GradSync

GradSync is a cross-machine distributed training framework for running a PyTorch model across a small cluster of heterogeneous machines. The top-level training script, `simple_vit.py`, uses the packages in `packages/` as a local library workspace to elect a cluster leader, partition model layers, and execute pipeline-parallel training over gRPC.

## Features

- Pipeline-parallel training across multiple machines.
- Automatic cluster election and ordered topology creation.
- Dynamic model-layer partitioning based on node capacity.
- gRPC transport for forward activations and backward gradients.
- Basic telemetry support for node metrics.
- Example MNIST transformer workload in `simple_vit.py`.

## Prerequisites

- Python 3.12 or newer.
- [uv](https://github.com/astral-sh/uv) for dependency and workspace management.
- Network connectivity between every machine in the training cluster.
- Open firewall access for the election port, training port, and telemetry ports used by the head node.

## Setup

Clone the repository and install the workspace dependencies:

```bash
git clone <repository-url>
cd gradsync
uv sync
```

You can run commands through `uv run` without manually activating the environment. If you prefer activating it:

```bash
# Linux, macOS, or WSL
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\activate
```

## Generating Protobuf Stubs

GradSync uses gRPC for both tensor transport and cluster orchestration. The generated `*_pb2.py` and `*_pb2_grpc.py` files are local build artifacts; new generated stubs are ignored by git, so generate them after cloning the repo, after running `uv sync`, and whenever a `.proto` file changes.

Run all commands from the repository root.

### macOS, Linux, WSL, or Git Bash

```bash
bash compile_proto.sh
```

### Windows PowerShell

If you are using native PowerShell instead of WSL or Git Bash, run:

```powershell
@("packages/comms/src/comms/proto", "packages/orchestrator/src/orchestrator/proto") | ForEach-Object { New-Item -ItemType Directory -Force $_ | Out-Null }

uv run python -m grpc_tools.protoc -I packages/comms/proto --python_out=packages/comms/src/comms/proto --grpc_python_out=packages/comms/src/comms/proto packages/comms/proto/tensor_service.proto

uv run python -m grpc_tools.protoc -I packages/orchestrator/proto --python_out=packages/orchestrator/src/orchestrator/proto --grpc_python_out=packages/orchestrator/src/orchestrator/proto packages/orchestrator/proto/cluster_service.proto

$patch = @'
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
'@

uv run python -c $patch
```

The Python patch step is required because `grpc_tools.protoc` generates imports such as `import tensor_service_pb2 as ...`, while this repo imports generated stubs as package modules and needs `from . import tensor_service_pb2 as ...`.

## Configuration

Cluster membership is defined in `cluster.json`:

```json
{
  "election_nodes": [
    "10.170.244.228:51234",
    "10.170.244.230:51234",
    "10.170.244.77:51234",
    "10.170.244.148:51234"
  ],
  "cluster_nodes": [
    "10.170.244.228:12345",
    "10.170.244.230:12345",
    "10.170.244.77:12345",
    "10.170.244.148:12345"
  ],
  "n_micro": 4
}
```

- `election_nodes` contains the orchestrator endpoints used for leader election.
- `cluster_nodes` contains the training endpoints used for pipeline traffic.
- `n_micro` controls the number of micro-batches used by the distributed pipeline.

Every machine must use the same `cluster.json`. The IP order must be identical for everyone, and the same machine must appear at the same index in both `election_nodes` and `cluster_nodes`.

## Running Distributed Training

Every machine that participates in the training cluster must run `simple_vit.py` with its own host IP:

```bash
uv run simple_vit.py --host_ip 10.170.244.148 --elec_port 51234 --train_port 12345 --config cluster.json
```

In this example, `10.170.244.148` is the host IP of the machine running the command. Each participant should replace it with that machine's own IP address while keeping the same ports and the same shared `cluster.json`.

When all configured machines are online:

1. The orchestrator waits for all peers.
2. The cluster elects a leader.
3. The leader builds the ordered topology and partitions model layers.
4. Worker nodes start serving their assigned model slice.
5. The head node begins training the MNIST transformer workload.

## Using the Packages as a Library

`simple_vit.py` is client code. It imports the workspace packages and wires them into a concrete training job:

```python
from pipeline import DistributedPipeline
```

The main public entry point for distributed training is `DistributedPipeline`. A client script provides a model builder, loss function, optimizer class, optimizer settings, local host IP, ports, and the path to the shared cluster config.

## Repository Structure

```text
gradsync/
├── simple_vit.py              # Distributed MNIST transformer example
├── simple_vit_no_dist.py      # Local non-distributed comparison script
├── cluster.json               # Shared cluster configuration
├── pyproject.toml             # Root uv workspace configuration
├── uv.lock                    # Locked dependency versions
└── packages/
    ├── common/                # Shared hardware and utility code
    ├── comms/                 # gRPC client/server transport for tensors
    ├── compression-lab/       # Tensor compression experiments and utilities
    ├── optimizer/             # Optimizer package placeholder
    ├── orchestrator/          # Cluster election, topology, and partition coordination
    ├── pipeline/              # DistributedPipeline and node runners
    └── telemetry/             # Telemetry server, client, tracker, and dashboard code
```

## Development Commands

```bash
# Sync all workspace dependencies
uv sync

# Run the distributed example
uv run simple_vit.py --host_ip <this-machine-ip> --elec_port 51234 --train_port 12345 --config cluster.json

# Regenerate gRPC protobuf stubs
bash compile_proto.sh

# Run tests
uv run pytest

# Add a dependency to a workspace package
uv add --package <package-name> <dependency-name>
```

Examples:

```bash
uv add --package pipeline torch
uv add --package orchestrator grpcio
```

## Troubleshooting

### A node waits forever for peers

Confirm every machine listed in `cluster.json` is running the same command, using its own `--host_ip`, and that the IP order in `cluster.json` is identical on every machine.

### Local endpoint is not listed in the config

The `--host_ip`, `--elec_port`, and `--train_port` values are combined into local endpoints. They must exactly match one entry in `election_nodes` and one entry in `cluster_nodes`.

### Port already in use

Stop the old process or update the port in both the command and `cluster.json`. Every participant must use the updated shared config.

### Firewall or network failures

Make sure machines can reach each other on the election port (`51234` in the example), training port (`12345` in the example), and telemetry ports (`8080` and `8081` on the head node).

### Missing protobuf modules

If you see an error such as `ModuleNotFoundError` for `*_pb2` or `*_pb2_grpc`, regenerate the stubs from the repo root:

```bash
bash compile_proto.sh
```

On native Windows PowerShell, use the commands in [Generating Protobuf Stubs](#generating-protobuf-stubs).

### Dependency issues

Recreate the environment if dependencies become inconsistent:

```bash
rm -rf .venv
uv sync
```

### Detected device=cpu (for machines with CUDA)

After `uv sync` run : 

```bash
uv <pip install command for the preferrec torch version>
```
ex:
```bash
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

