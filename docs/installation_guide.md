# Fighter Jet SDK Installation Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Dependencies](#dependencies)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Installation](#advanced-installation)
8. [Docker Deployment](#docker-deployment)
9. [Development Installation](#development-installation)

## System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+), macOS 10.15+, Windows 10+
- **Python**: 3.8 or higher
- **Memory**: 8 GB RAM minimum, 16 GB recommended
- **Storage**: 10 GB free space minimum, 50 GB recommended for full datasets
- **CPU**: Multi-core processor recommended for parallel processing

### Recommended Requirements

- **Operating System**: Linux (Ubuntu 22.04 LTS)
- **Python**: 3.10 or higher
- **Memory**: 32 GB RAM or higher
- **Storage**: 100 GB+ SSD storage
- **CPU**: 16+ cores for optimal performance
- **GPU**: NVIDIA GPU with CUDA support (optional, for accelerated computations)

### External Dependencies

- **OpenFOAM**: Version 9 or higher (for CFD analysis)
- **ParaView**: For visualization (optional)
- **Git**: For version control
- **Build Tools**: GCC/Clang compiler, Make, CMake

## Installation Methods

### Method 1: Package Installation (Recommended)

```bash
# Install from PyPI (when available)
pip install fighter-jet-sdk

# Or install from wheel file
pip install fighter_jet_sdk-0.1.0-py3-none-any.whl
```

### Method 2: Source Installation

```bash
# Clone the repository
git clone https://github.com/your-org/fighter-jet-sdk.git
cd fighter-jet-sdk

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Method 3: Conda Installation

```bash
# Create conda environment
conda create -n fighter-jet-sdk python=3.10
conda activate fighter-jet-sdk

# Install from conda-forge (when available)
conda install -c conda-forge fighter-jet-sdk

# Or install dependencies and then pip install
conda install numpy scipy matplotlib pyyaml
pip install fighter-jet-sdk
```

## Dependencies

### Python Dependencies

The SDK requires the following Python packages:

```bash
# Core scientific computing
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0

# Data handling
pandas>=1.3.0
h5py>=3.6.0
pyyaml>=6.0

# Parallel processing
multiprocessing-logging>=0.3.0
psutil>=5.8.0

# Visualization
plotly>=5.0.0
vtk>=9.0.0

# Machine learning (optional)
scikit-learn>=1.0.0
torch>=1.10.0

# Development tools
pytest>=6.0.0
black>=22.0.0
flake8>=4.0.0
```

### System Dependencies

#### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install build tools
sudo apt install build-essential cmake git

# Install OpenFOAM
sudo apt install openfoam9

# Install additional libraries
sudo apt install libhdf5-dev libopenmpi-dev

# Install Python development headers
sudo apt install python3-dev python3-pip
```

#### CentOS/RHEL/Fedora

```bash
# Install build tools
sudo dnf install gcc gcc-c++ cmake git make

# Install OpenFOAM (may require additional repositories)
sudo dnf install openfoam

# Install additional libraries
sudo dnf install hdf5-devel openmpi-devel

# Install Python development headers
sudo dnf install python3-devel python3-pip
```

#### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake git hdf5 open-mpi

# Install OpenFOAM
brew install openfoam

# Install Python
brew install python@3.10
```

#### Windows

For Windows, we recommend using Windows Subsystem for Linux (WSL2) or Docker:

```powershell
# Install WSL2
wsl --install

# Then follow Linux installation instructions within WSL2
```

## Configuration

### Initial Setup

1. **Create configuration directory**:
```bash
mkdir -p ~/.fighter_jet_sdk
```

2. **Initialize default configuration**:
```bash
fighter-jet-sdk config init
```

3. **Verify configuration**:
```bash
fighter-jet-sdk config validate
```

### Environment Variables

Set the following environment variables for optimal performance:

```bash
# Add to ~/.bashrc or ~/.zshrc

# OpenFOAM environment
source /opt/openfoam9/etc/bashrc

# Fighter Jet SDK settings
export FIGHTER_JET_SDK_HOME="$HOME/.fighter_jet_sdk"
export FIGHTER_JET_SDK_DATA_DIR="$HOME/.fighter_jet_sdk/data"
export FIGHTER_JET_SDK_CACHE_DIR="$HOME/.fighter_jet_sdk/cache"

# Performance settings
export OMP_NUM_THREADS=8  # Adjust based on your CPU cores
export OPENBLAS_NUM_THREADS=8

# Optional: CUDA settings for GPU acceleration
export CUDA_VISIBLE_DEVICES=0
```

### Configuration File

Create or edit the configuration file at `~/.fighter_jet_sdk/config.yaml`:

```yaml
# Fighter Jet SDK Configuration

# Logging settings
log_level: "INFO"
log_file: "~/.fighter_jet_sdk/logs/sdk.log"

# Performance settings
parallel_processing: true
max_threads: null  # Auto-detect
cache_enabled: true
cache_size_mb: 2048

# Data directories
data_directory: "~/.fighter_jet_sdk/data"
backup_enabled: true
backup_interval_hours: 24

# Engine configurations
engines:
  design:
    module_library_path: "~/.fighter_jet_sdk/modules"
    validation_strict: true
  
  materials:
    database_path: "~/.fighter_jet_sdk/materials.db"
    simulation_precision: "high"
  
  propulsion:
    cfd_solver: "openfoam"
    thermal_analysis: true
  
  aerodynamics:
    cfd_mesh_density: "medium"
    turbulence_model: "k-omega-sst"
  
  manufacturing:
    cost_database_path: "~/.fighter_jet_sdk/costs.db"
    quality_standards: "aerospace"
```

## Verification

### Basic Verification

```bash
# Check installation
fighter-jet-sdk --version

# Verify configuration
fighter-jet-sdk config show

# Test basic functionality
fighter-jet-sdk examples --category basic
```

### Comprehensive Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_basic_functionality.py
python -m pytest tests/test_engine_integration.py

# Run performance benchmarks
fighter-jet-sdk workflow benchmark --reference f22
```

### Interactive Verification

```bash
# Start interactive mode
fighter-jet-sdk interactive

# In interactive mode, test each engine:
fighter-jet-sdk> status
fighter-jet-sdk> design list
fighter-jet-sdk> materials list
fighter-jet-sdk> config show
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` when importing SDK components

**Solution**:
```bash
# Reinstall with dependencies
pip install --force-reinstall fighter-jet-sdk

# Or install missing dependencies
pip install -r requirements.txt
```

#### 2. OpenFOAM Not Found

**Problem**: CFD analysis fails with OpenFOAM errors

**Solution**:
```bash
# Verify OpenFOAM installation
which simpleFoam

# Source OpenFOAM environment
source /opt/openfoam9/etc/bashrc

# Add to shell profile for persistence
echo "source /opt/openfoam9/etc/bashrc" >> ~/.bashrc
```

#### 3. Permission Errors

**Problem**: Permission denied when creating files or directories

**Solution**:
```bash
# Fix permissions for SDK directory
chmod -R 755 ~/.fighter_jet_sdk

# Create directories with proper permissions
mkdir -p ~/.fighter_jet_sdk/{data,cache,logs,modules}
```

#### 4. Memory Issues

**Problem**: Out of memory errors during analysis

**Solution**:
```bash
# Reduce cache size in configuration
fighter-jet-sdk config set cache_size_mb 512

# Use single precision for simulations
fighter-jet-sdk config set simulation_precision single

# Limit parallel threads
fighter-jet-sdk config set max_threads 4
```

#### 5. Performance Issues

**Problem**: Slow execution times

**Solution**:
```bash
# Enable parallel processing
fighter-jet-sdk config set parallel_processing true

# Increase cache size (if memory allows)
fighter-jet-sdk config set cache_size_mb 4096

# Use coarser analysis settings
fighter-jet-sdk config set engines.aerodynamics.cfd_mesh_density coarse
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Set debug log level
export FIGHTER_JET_SDK_LOG_LEVEL=DEBUG

# Or use command line option
fighter-jet-sdk --log-level DEBUG <command>
```

### Log Files

Check log files for detailed error information:

```bash
# View recent logs
tail -f ~/.fighter_jet_sdk/logs/sdk.log

# Search for errors
grep -i error ~/.fighter_jet_sdk/logs/sdk.log

# View specific engine logs
ls ~/.fighter_jet_sdk/logs/
```

## Advanced Installation

### Custom Installation Paths

```bash
# Install to custom location
pip install --prefix=/opt/fighter-jet-sdk fighter-jet-sdk

# Set custom data directory
export FIGHTER_JET_SDK_DATA_DIR="/data/fighter-jet-sdk"

# Use custom configuration file
fighter-jet-sdk --config /path/to/custom/config.yaml
```

### Multi-User Installation

For system-wide installation:

```bash
# Install system-wide
sudo pip install fighter-jet-sdk

# Create shared data directory
sudo mkdir -p /opt/fighter-jet-sdk/data
sudo chmod 755 /opt/fighter-jet-sdk/data

# Create system configuration
sudo mkdir -p /etc/fighter-jet-sdk
sudo cp config.yaml /etc/fighter-jet-sdk/
```

### High-Performance Computing (HPC)

For HPC environments:

```bash
# Load required modules
module load python/3.10
module load openmpi/4.1
module load openfoam/9

# Install in user space
pip install --user fighter-jet-sdk

# Configure for cluster usage
fighter-jet-sdk config set parallel_processing true
fighter-jet-sdk config set max_threads $SLURM_CPUS_PER_TASK
```

## Docker Deployment

### Using Pre-built Image

```bash
# Pull the official image
docker pull fighter-jet-sdk:latest

# Run interactive container
docker run -it --rm \
  -v $(pwd):/workspace \
  -v ~/.fighter_jet_sdk:/root/.fighter_jet_sdk \
  fighter-jet-sdk:latest

# Run specific command
docker run --rm \
  -v $(pwd):/workspace \
  fighter-jet-sdk:latest \
  fighter-jet-sdk design create --name TestFighter
```

### Building Custom Image

Create a `Dockerfile`:

```dockerfile
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    build-essential cmake git \
    openfoam9 \
    libhdf5-dev libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

# Install Fighter Jet SDK
COPY . /tmp/fighter-jet-sdk/
RUN cd /tmp/fighter-jet-sdk && pip3 install .

# Setup environment
RUN echo "source /opt/openfoam9/etc/bashrc" >> /root/.bashrc

# Create working directory
WORKDIR /workspace

# Default command
CMD ["fighter-jet-sdk", "--help"]
```

Build and run:

```bash
# Build image
docker build -t fighter-jet-sdk:custom .

# Run container
docker run -it --rm fighter-jet-sdk:custom
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  fighter-jet-sdk:
    image: fighter-jet-sdk:latest
    volumes:
      - ./workspace:/workspace
      - ./config:/root/.fighter_jet_sdk
    environment:
      - FIGHTER_JET_SDK_LOG_LEVEL=INFO
    command: fighter-jet-sdk interactive
    
  jupyter:
    image: fighter-jet-sdk:latest
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/workspace
    command: jupyter lab --ip=0.0.0.0 --allow-root
```

## Development Installation

For developers contributing to the SDK:

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/fighter-jet-sdk.git
cd fighter-jet-sdk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

```bash
# Testing
pytest>=6.0.0
pytest-cov>=3.0.0
pytest-mock>=3.6.0

# Code quality
black>=22.0.0
flake8>=4.0.0
isort>=5.10.0
mypy>=0.950

# Documentation
sphinx>=4.5.0
sphinx-rtd-theme>=1.0.0

# Development tools
pre-commit>=2.17.0
tox>=3.24.0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fighter_jet_sdk

# Run specific test file
pytest tests/test_design_engine.py

# Run tests in parallel
pytest -n auto
```

### Code Quality

```bash
# Format code
black fighter_jet_sdk/
isort fighter_jet_sdk/

# Lint code
flake8 fighter_jet_sdk/
mypy fighter_jet_sdk/

# Run all quality checks
pre-commit run --all-files
```

---

This installation guide provides comprehensive instructions for installing and configuring the Fighter Jet SDK across different platforms and use cases. For additional support, refer to the troubleshooting section or contact the development team.