# gradsync

A cross-OS distributed machine learning training framework that enables pipeline parallelism across heterogeneous systems (Windows, macOS, Linux) with aggressive data compression for network optimization.

## Project Overview

GRADSYNC implements multi-system ML training algorithms using gRPC and reliable network protocols to enable lightweight, cross-OS distributed training on consumer hardware. The system uses pipeline parallelism where models are split sequentially across systems, maximizing throughput by minimizing communication overhead.

## Prerequisites
- [uv](https://github.com/astral-sh/uv) - A fast Python package installer and resolver

## Setup

### Project Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd gradsync
```

2. Install project dependencies:
```bash
uv sync
```

3. Source the env 
```bash
# in Linux / WSL
source ./.venv/bin/activate 
```

```powershell
.\.venv\Scripts\activate
```


This will:
- Download and install all dependencies from `pyproject.toml`
- Create lock files to ensure reproducible installs
- Install the project in editable mode (if configured)

### Running the Project

After setting up the environment, you can run the main script:

```bash
 uv run main.py --role tail (run tail first)
 uv run main.py --role head --target_ip 127.0.0.1
```

## Common UV Commands

Here are the essential uv commands you'll use in this project:

```bash
# Create virtual environment
uv venv

# Install/sync all dependencies from pyproject.toml
uv sync

# Add a new dependency
uv add <dependency-name>

# Add dependency to any sub package (in /packages folder)
uv add --package <sub-pipeline> <depenendecy-name>
# where dependency can other sub package as well
```

## Project Structure

```
gradsync/
├── main.py              # Main entry point
├── pyproject.toml       # Project configuration and dependencies
├── packages/            # Core packages directory
│   ├── comm/           # Communication layer
│   ├── compression-lab/ # Tensor compression algorithms
│   ├── optimizer/      # Optimization algorithms
│   ├── telemetry/      # Monitoring and metrics
│   └── pipeline/       # Pipeline parallelism implementation
├── uv.lock             # Locked dependency versions (auto-generated)
└── .python-version     # Python version specification
```

## Troubleshooting

### Virtual environment issues
If you encounter issues with the virtual environment:
```bash
# Remove existing virtual environment
rm -rf .venv

# Recreate and reinstall
uv sync
```

### Dependency conflicts
```bash
# Clear uv cache
uv cache clean

# Reinstall with fresh cache
uv sync --reinstall
```

### Platform-specific notes
- On Windows WSL, ensure you're using the Linux version of uv
- On macOS with Apple Silicon, uv will automatically handle architecture-specific packages
- Cross-platform compatibility is handled automatically