from .tensor_compression_new import TensorCompressor, CompressionType


# Export main class and utilities
__all__ = [
    'TensorCompressor',
    'CompressionType',
    'get_optimal_compression',
    'validate_tensor_compression'
]
