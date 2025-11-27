from .architectures import (
    create_model,
    create_model_with_architecture,
    get_architecture_specific_param_grid,
    get_supported_architectures,
)

from .model_adapters import (
    adapt_model_for_grayscale,
    verify_grayscale_adaptation,
    get_adaptation_info
)

__all__ = [
    # Architectures
    'create_model',
    'create_model_with_architecture',
    'get_architecture_specific_param_grid',
    'get_supported_architectures',

    # Model adapters
    'adapt_model_for_grayscale',
    'verify_grayscale_adaptation',
    "get_adaptation_info"
]