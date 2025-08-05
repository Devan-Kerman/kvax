# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
```bash
# Install development dependencies
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_attention.py

# Run specific test function
pytest tests/test_attention.py::test_function_name
```

### Code Quality
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run individual formatters/linters
black .
isort .
flake8 .
```

### Benchmarking
```bash
# Basic forward pass benchmark
python3 benchmarks.py mha

# Forward pass with document segments and padding
python3 benchmarks.py mha --num-segments 3 --num-pad-tokens 1000

# Forward+backward benchmark
python3 benchmarks.py mha_bwd --num-segments 3 --num-pad-tokens 1000

# Context parallelism benchmark
python3 benchmarks.py mha_cp

# Show attention mask visualization
python3 benchmarks.py mha --num-segments 3 --show-attention-mask
```

## Architecture Overview

### Core Package Structure
- **kvax.ops**: Flash attention implementations
  - `flash_attention_triton.py`: Main Triton-based implementation (original API)
  - `flash_attention_clean.py`: Simplified API without context managers
  - `flash_attention_cudnn.py`: CuDNN reference implementation for benchmarking
  - `mask_creator.py`: Block-wise attention mask generation
- **kvax.utils**: Utilities and configuration
  - `common.py`: Configuration dataclasses and constants (FlashAttentionParamsConfig, PADDING_SEGMENT_ID)
  - `specs.py`: Sharding specification context manager
  - `permutation.py`: Token permutation for context parallelism load balancing
  - `sharding.py`: JAX sharding utilities

### Key Concepts

#### Two API Styles
1. **Clean API (Recommended)**: Direct function calls without context managers
   - Import from `kvax.ops.flash_attention_clean`
   - Mesh and sharding specs passed as function arguments
   - Works seamlessly on single device (no mesh/specs needed) or multi-device
   
2. **Original API**: Uses context managers for mesh and sharding configuration
   - Import from `kvax.ops` 
   - Requires `attention_specs()` context manager
   - Must be used within JAX mesh context

#### Flash Attention Implementation
- Built on Triton kernels for GPU optimization
- Block-wise attention mask computation to handle document boundaries efficiently
- Supports grouped query attention (GQA) with memory-optimized backward pass
- GPU-specific parameter tuning (H100 optimizations available)

#### Context Parallelism
- All-gather based approach following Llama 3 training methodology
- Token permutation for load balancing across GPUs in causal mask scenarios
- Balances computation across sequence dimension while maintaining batch sharding

#### Document Masks and Padding
- Uses `PADDING_SEGMENT_ID = -1` to mark padding tokens in segment_ids tensors
- Block-wise mask storage: `3 * 4 * batch_size * seq_len // block_size * 4 bytes`
- Skips computation on padding-only blocks for efficiency

### Development Notes

#### Dependencies
- **Core**: JAX (>=0.4.34), JAX-Triton (>=0.2.0), Triton (>=3.1), Chex (>=0.1.85)
- **Dev**: pytest, pre-commit, click
- **Compatibility**: Ensure Triton and JAX-Triton versions are compatible (tested with triton==3.1, jax-triton==0.2.0)

#### Testing Patterns
- Tests compare Triton implementation against CuDNN reference
- Forward/backward correctness verification via `check_outputs_fwd/bwd`
- Multi-device testing with various sharding configurations
- Segment mask and causal mask validation

#### Performance Considerations
- `FlashAttentionParamsConfig` controls kernel parameters (block sizes, warps, stages)
- Larger block sizes improve performance but require more GPU memory
- H100-specific optimizations available via device detection
- Context parallelism load balancing can be toggled via `permute_tokens_for_load_balance`

#### Code Quality Standards
- Black formatting (line length defaults)
- isort import sorting
- flake8 linting
- Pre-commit hooks enforce standards automatically