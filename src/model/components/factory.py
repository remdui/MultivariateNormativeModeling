"""Factory module for creating encoder and decoder instances."""

from model.components.base_decoder import BaseDecoder
from model.components.base_encoder import BaseEncoder
from model.components.mlp.decoder import MLPDecoder
from model.components.mlp.encoder import MLPEncoder

# Mapping for available encoders
ENCODER_MAPPING: dict[str, type[BaseEncoder]] = {
    "mlp": MLPEncoder,
}

# Mapping for available decoders
DECODER_MAPPING: dict[str, type[BaseDecoder]] = {
    "mlp": MLPDecoder,
}


def get_encoder(
    encoder_type: str, input_dim: int, hidden_dims: list[int], latent_dim: int
) -> BaseEncoder:
    """Factory method to get the encoder based on config.

    Args:
        encoder_type (str): The type of encoder (e.g., 'mlp').
        input_dim (int): Dimension of the input.
        hidden_dims (list[int]): Dimensions of the hidden layers.
        latent_dim (int): Dimension of the latent space.

    Returns:
        BaseEncoder: An instance of the specified encoder.

    Raises:
        ValueError: If the encoder type is not supported.
    """
    encoder_class = ENCODER_MAPPING.get(encoder_type.lower())
    if not encoder_class:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    return encoder_class(input_dim, hidden_dims, latent_dim)


def get_decoder(
    decoder_type: str, latent_dim: int, hidden_dims: list[int], output_dim: int
) -> BaseDecoder:
    """Factory method to get the decoder based on config.

    Args:
        decoder_type (str): The type of decoder (e.g., 'mlp').
        latent_dim (int): Dimension of the latent space.
        hidden_dims (list[int]): Dimensions of the hidden layers.
        output_dim (int): Dimension of the output.

    Returns:
        BaseDecoder: An instance of the specified decoder.

    Raises:
        ValueError: If the decoder type is not supported.
    """
    decoder_class = DECODER_MAPPING.get(decoder_type.lower())
    if not decoder_class:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
    return decoder_class(latent_dim, hidden_dims, output_dim)
