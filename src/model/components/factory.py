"""Factory methods to get encoder and decoder based on config."""

from model.components.mlp.decoder import MLPDecoder
from model.components.mlp.encoder import MLPEncoder


def get_encoder(encoder_type, input_dim, hidden_dims, latent_dim):
    """Factory method to get the encoder based on config."""
    if encoder_type == "mlp":
        return MLPEncoder(input_dim, hidden_dims, latent_dim)
    raise ValueError(f"Unknown encoder type: {encoder_type}")


def get_decoder(decoder_type, latent_dim, hidden_dims, output_dim):
    """Factory method to get the decoder based on config."""
    if decoder_type == "mlp":
        return MLPDecoder(latent_dim, hidden_dims, output_dim)
    raise ValueError(f"Unknown decoder type: {decoder_type}")
