"""Factory module for creating encoder and decoder instances."""

from typing import Any

from model.components.decoder.base_decoder import BaseDecoder
from model.components.decoder.impl.mlp_decoder import MLPDecoder
from model.components.encoder.base_encoder import BaseEncoder
from model.components.encoder.impl.mlp_encoder import MLPEncoder
from model.components.encoder.impl.mlp_encoder_ba import MLPEncoderBA

# Type aliases for clarity
EncoderClass = type[BaseEncoder]
DecoderClass = type[BaseDecoder]

# Mapping for available encoders (private)
_ENCODER_MAPPING: dict[str, EncoderClass] = {
    "mlp": MLPEncoder,
    "mlp_ba": MLPEncoderBA,
}

# Mapping for available decoders (private)
_DECODER_MAPPING: dict[str, DecoderClass] = {
    "mlp": MLPDecoder,
}


def get_encoder(encoder_type: str, *args: Any, **kwargs: Any) -> BaseEncoder:
    """
    Factory method to create an encoder instance based on configuration.

    Args:
        encoder_type (str): The type of encoder (e.g., 'mlp'). The lookup is case-insensitive.
        *args: Positional arguments for the encoder's constructor.
        **kwargs: Keyword arguments for the encoder's constructor.

    Returns:
        BaseEncoder: An instance of the specified encoder.

    Raises:
        ValueError: If the encoder type is not supported.
    """
    encoder_class = _ENCODER_MAPPING.get(encoder_type.lower())
    if encoder_class is None:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    return encoder_class(*args, **kwargs)


def get_decoder(decoder_type: str, *args: Any, **kwargs: Any) -> BaseDecoder:
    """
    Factory method to create a decoder instance based on configuration.

    Args:
        decoder_type (str): The type of decoder (e.g., 'mlp'). The lookup is case-insensitive.
        *args: Positional arguments for the decoder's constructor.
        **kwargs: Keyword arguments for the decoder's constructor.

    Returns:
        BaseDecoder: An instance of the specified decoder.

    Raises:
        ValueError: If the decoder type is not supported.
    """
    decoder_class = _DECODER_MAPPING.get(decoder_type.lower())
    if decoder_class is None:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
    return decoder_class(*args, **kwargs)
