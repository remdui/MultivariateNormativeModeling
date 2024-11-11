"""Factory module for creating encoder and decoder instances."""

from typing import Any

from model.components.decoder.base_decoder import BaseDecoder
from model.components.decoder.impl.mlp_decoder import MLPDecoder
from model.components.encoder.base_encoder import BaseEncoder
from model.components.encoder.impl.mlp_encoder import MLPEncoder

# Mapping for available encoders
ENCODER_MAPPING: dict[str, type[BaseEncoder]] = {
    "mlp": MLPEncoder,
}

# Mapping for available decoders
DECODER_MAPPING: dict[str, type[BaseDecoder]] = {
    "mlp": MLPDecoder,
}


def get_encoder(encoder_type: str, *args: Any, **kwargs: Any) -> BaseEncoder:
    """Factory method to get the encoder based on config.

    Args:
        encoder_type (str): The type of encoder (e.g., 'mlp').
        *args: Additional arguments specific to the encoder.
        **kwargs: Additional parameters specific to the encoder

    Returns:
        BaseEncoder: An instance of the specified encoder.

    Raises:
        ValueError: If the encoder type is not supported.
    """
    encoder_class = ENCODER_MAPPING.get(encoder_type.lower())
    if not encoder_class:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    return encoder_class(*args, **kwargs)


def get_decoder(decoder_type: str, *args: Any, **kwargs: Any) -> BaseDecoder:
    """Factory method to get the decoder based on config.

    Args:
        decoder_type (str): The type of decoder (e.g., 'mlp').
        *args: Additional arguments specific to the decoder.
        **kwargs: Additional parameters specific to the decoder.

    Returns:
        BaseDecoder: An instance of the specified decoder.

    Raises:
        ValueError: If the decoder type is not supported.
    """
    decoder_class = DECODER_MAPPING.get(decoder_type.lower())
    if not decoder_class:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
    return decoder_class(*args, **kwargs)
