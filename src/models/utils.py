"""Neural network components for the diffusion model.

This module contains the building blocks used in the U-Net architecture:
- Time embeddings with sinusoidal encoding
- Residual blocks with time and class conditioning
- Multi-head self-attention for global context
- Group normalization for stable training
"""

import math

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn


def create_mask(
    image_size: tuple[int, int], box_size: int, position: tuple[int, int]
) -> torch.Tensor:
    """Create a binary mask for inpainting."""
    mask = torch.zeros((1, 1, *image_size))
    x, y = position
    mask[:, :, y : y + box_size, x : x + box_size] = 1
    return mask


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding.

    Mathematical Background:
    Uses the same positional encoding as in the Transformer paper:

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This creates a unique, continuous embedding for each timestep that the
    model can learn to interpret. The sinusoidal pattern allows the model
    to generalize to unseen timesteps during sampling.

    Args:
        dim: Embedding dimension

    """

    def __init__(self, dim: int) -> None:
        """Initialize the SinusoidalPositionEmbeddings layer."""
        super().__init__()
        self.dim = dim

    def forward(self, timestep: Int[Tensor, "batch"], /) -> Float[Tensor, "batch dim"]:  # type: ignore[name-defined]
        """Generate sinusoidal embeddings for given timesteps.

        Args:
            timestep: Timesteps, shape (batch,)

        Returns:
            Sinusoidal embeddings, shape (batch, dim)

        """
        device = timestep.device
        half_dim = self.dim // 2

        # Create frequency scaling: 10000^(2i/d_model)
        _embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -_embeddings)

        # Apply frequencies to timesteps
        embeddings = timestep[:, None] * embeddings[None, :]

        # Concatenate sin and cos
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ResnetBlock(nn.Module):
    """Residual block with time and class conditioning.

    Mathematical Background:
    Implements the residual connection: h = x + F(x, t, y)
    where F incorporates:
    - Time information through learned embeddings
    - Class information for conditional generation
    - Two conv layers with group normalization

    The residual connection helps with gradient flow and allows
    the model to learn identity mappings when beneficial.

    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        time_emb_dim: Time embedding dimension
        num_classes: Number of class labels
        dropout: Dropout probability

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the ResnetBlock layer."""
        super().__init__()

        # First conv path
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Time conditioning
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

        # Class conditioning
        self.class_mlp = nn.Sequential(nn.SiLU(), nn.Linear(num_classes, out_channels))

        # Second conv path
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection projection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(
        self,
        x: Float[Tensor, "batch channels height width"],
        t_emb: Float[Tensor, "batch time_dim"],
        y_emb: Float[Tensor, "batch class_dim"],
    ) -> Float[Tensor, "batch channels height width"]:
        """Forward pass with time and class conditioning.

        Args:
            x: Input feature maps
            t_emb: Time embeddings
            y_emb: Class embeddings

        Returns:
            Output feature maps with residual connection

        """
        # First conv with normalization and activation
        h = self.conv1(F.silu(self.norm1(x)))

        # Add time conditioning (broadcast to spatial dimensions)
        h += self.time_mlp(t_emb)[:, :, None, None]

        # Add class conditioning (broadcast to spatial dimensions)
        h += self.class_mlp(y_emb)[:, :, None, None]

        # Second conv with dropout
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))

        # Residual connection
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block for capturing global dependencies.

    Mathematical Background:
    Implements multi-head self-attention:

    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    This allows the model to attend to global spatial relationships,
    which is crucial for generating coherent images. Unlike local
    convolutions, attention can relate distant pixels directly.

    Args:
        channels: Number of input channels
        num_heads: Number of attention heads

    """

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        """Initialize the AttentionBlock layer."""
        super().__init__()
        if channels % num_heads != 0:
            msg = "channels must be divisible by num_heads"
            raise ValueError(msg)

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        # Layer norm for stable training
        self.norm = nn.GroupNorm(8, channels)

        # Linear projections for Q, K, V
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(
        self, x: Float[Tensor, "batch channels h w"]
    ) -> Float[Tensor, "batch channels h w"]:
        """Apply multi-head self-attention.

        Args:
            x: Input feature maps, shape (batch, channels, height, width)

        Returns:
            Output feature maps with attention applied

        """
        b, c, h, w = x.shape

        # Normalize input
        x_norm = self.norm(x)

        # Compute Q, K, V
        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = (t.view(b, self.num_heads, self.head_dim, h * w) for t in qkv)

        # Compute attention scores: QK^T / √d_k
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values: Attention(Q,K,V) = softmax(QK^T/√d_k)V
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.contiguous().view(b, c, h, w)

        # Output projection and residual connection
        return x + self.to_out(out)


class Downsample(nn.Module):
    """Downsampling layer that reduces spatial resolution by 2x.

    Uses stride-2 convolution for learnable downsampling,
    which is more flexible than fixed pooling operations.
    """

    def __init__(self, channels: int) -> None:
        """Initialize the Downsample layer."""
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(
        self, x: Float[Tensor, "batch n h w"]
    ) -> Float[Tensor, "batch m h1 w1"]:
        """Forward pass with stride-2 convolution."""
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer that increases spatial resolution by 2x.

    Uses transposed convolution for learnable upsampling,
    followed by a regular conv to reduce artifacts.
    """

    def __init__(self, channels: int) -> None:
        """Initialize the Upsample layer."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            channels, channels, 4, stride=2, padding=1
        )
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self, x: Float[Tensor, "batch n h w"]
    ) -> Float[Tensor, "batch m h1 w1"]:
        """Forward pass with transposed convolution and regular conv."""
        x = self.conv_transpose(x)
        return self.conv(x)
