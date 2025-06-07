"""U-Net architecture for diffusion model noise prediction.

This module implements the core U-Net model used for predicting noise
in the DDPM framework. The architecture includes:
- Time and class conditioning
- Skip connections for multi-scale feature fusion
- Self-attention for global context
"""

from dataclasses import field

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic.dataclasses import dataclass
from rich.console import Console
from torch import Tensor, nn
from torchsummary import summary

from config import ModelConfig
from src.models.utils import (
    AttentionBlock,
    Downsample,
    ResnetBlock,
    SinusoidalPositionEmbeddings,
    Upsample,
)

console = Console()


@dataclass
class UNet(nn.Module, ModelConfig):
    """U-Net architecture for noise prediction in diffusion models.

    Mathematical Background:
    The model learns to predict the noise ε added at timestep t:
    ε_θ(x_t, t, y) ≈ ε

    where x_t = √α̅_t x_0 + √(1-α̅_t) ε

    Architecture Design:
    - Encoder: Progressive downsampling with skip connections
    - Bottleneck: High-level feature processing
    - Decoder: Progressive upsampling with skip connection fusion
    - Conditioning: Time and class embeddings injected at each level

    The U-Net structure allows the model to process features at multiple
    scales while preserving fine-grained details through skip connections.
    """

    mid_blocks: nn.ModuleList = field(init=False)  # Bottleneck blocks
    encoder_blocks: nn.ModuleList = field(init=False)  # Encoder blocks
    decoder_blocks: nn.ModuleList = field(init=False)  # Decoder blocks
    downsample_blocks: nn.ModuleList = field(init=False)  # Downsampling blocks
    upsample_blocks: nn.ModuleList = field(init=False)  # Upsampling blocks

    conv_in: nn.Conv2d = field(init=False)  # Input convolution
    time_embed: nn.Sequential = field(init=False)  # Time embedding network
    class_embed: nn.Embedding = field(init=False)  # Class embedding

    def __post_init__(self) -> None:
        """Initialize the U-Net model."""
        super().__init__()
        self.build_model()

    def __hash__(self) -> int:
        """Generate a hash for the UNet instance."""
        return hash(
            tuple(getattr(self, field) for field in ModelConfig.__dataclass_fields__)
        )

    def __eq__(self, other: object) -> bool:
        """Check if two UNet instances are equal."""
        if not isinstance(other, UNet):
            return False
        return hash(self) == hash(other)

    def build_model(self) -> None:
        """Build the U-Net model architecture."""
        # Calculate channel counts for each level
        channels = [self.base_channels * mult for mult in self.channel_mults]

        # Time embedding network
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim // 4),
            nn.Linear(self.time_emb_dim // 4, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        # Class embedding (converts class index to one-hot-like embedding)
        self.class_embed = nn.Embedding(self.num_classes, self.time_emb_dim)

        # Initial convolution
        self.conv_in = nn.Conv2d(self.img_channels, self.base_channels, 3, padding=1)

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        in_ch = self.base_channels
        resolution = self.img_size

        for level, out_ch in enumerate(channels):
            # ResNet blocks for this level
            blocks = nn.ModuleList()

            for _ in range(self.num_res_blocks):
                blocks.append(
                    ResnetBlock(
                        in_ch,
                        out_ch,
                        self.time_emb_dim,
                        self.time_emb_dim,
                        self.dropout,
                    )
                )
                in_ch = out_ch

                # Add attention if at specified resolution
                if resolution in self.attention_resolutions:
                    blocks.append(AttentionBlock(out_ch, self.num_heads))

            self.encoder_blocks.append(blocks)

            # Downsample (except for last level)
            if level < len(channels) - 1:
                self.downsample_blocks.append(Downsample(out_ch))
                resolution //= 2
            else:
                self.downsample_blocks.append(nn.Identity())

        # Bottleneck
        mid_ch = channels[-1]
        self.mid_blocks = nn.ModuleList(
            [
                ResnetBlock(
                    mid_ch, mid_ch, self.time_emb_dim, self.time_emb_dim, self.dropout
                ),
                AttentionBlock(mid_ch, self.num_heads),
                ResnetBlock(
                    mid_ch, mid_ch, self.time_emb_dim, self.time_emb_dim, self.dropout
                ),
            ]
        )

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        # Reverse the channel list for decoder
        channels_reversed = list(reversed(channels))

        for level, out_ch in enumerate(channels_reversed):
            # For decoder, input channels include skip connections
            in_ch = out_ch
            # Account for skip connection channels
            skip_ch = out_ch
            total_in_ch = in_ch + skip_ch

            # ResNet blocks for this level
            blocks = nn.ModuleList()

            for i in range(
                self.num_res_blocks + 1
            ):  # +1 for skip connection processing
                _out_ch = out_ch // 2 if i == self.num_res_blocks else out_ch
                if i == 0:
                    # First block processes concatenated features
                    blocks.append(
                        ResnetBlock(
                            total_in_ch,
                            out_ch,
                            self.time_emb_dim,
                            self.time_emb_dim,
                            self.dropout,
                        )
                    )
                else:
                    blocks.append(
                        ResnetBlock(
                            out_ch,
                            _out_ch,
                            self.time_emb_dim,
                            self.time_emb_dim,
                            self.dropout,
                        )
                    )

                # Add attention if at specified resolution
                resolution = self.img_size // (2 ** (len(channels) - 1 - level))
                if resolution in self.attention_resolutions:
                    blocks.append(AttentionBlock(_out_ch, self.num_heads))

            self.decoder_blocks.append(blocks)

            # Upsample (except for last level)
            if level < len(channels) - 1:
                self.upsample_blocks.append(Upsample(out_ch))
            else:
                self.upsample_blocks.append(nn.Identity())

        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, self.base_channels // 2),
            nn.SiLU(),
            nn.Conv2d(self.base_channels // 2, self.img_channels, 3, padding=1),
        )

    def forward(
        self,
        x: Float[Tensor, "batch channels height width"],
        t: Int[Tensor, "batch"],  # type: ignore[name-defined]
        y: Int[Tensor, "batch"],  # type: ignore[name-defined]
    ) -> Float[Tensor, "batch n height width"]:
        """Forward pass of the U-Net.

        Mathematical Background:
        The network computes ε_θ(x_t, t, y), the predicted noise that was
        added to the clean image x_0 to create the noisy image x_t at timestep t
        for class y.

        The loss function during training is:
        L = ||ε - ε_θ(x_t, t, y)||²

        Args:
            x: Noisy image x_t, shape (batch, channels, height, width)
            t: Timestep, shape (batch,) with values in [0, T-1]
            y: Class label, shape (batch,) with values in [0, num_classes-1]

        Returns:
            Predicted noise ε_θ(x_t, t, y), same shape as input x

        """
        # Generate embeddings
        t_emb = self.time_embed(t.squeeze().long())  # (batch, time_emb_dim)
        y_emb = self.class_embed(y.squeeze().long())  # (batch, time_emb_dim)

        # Initial convolution
        x = self.conv_in(x)  # (batch, base_channels, height, width)

        # Store skip connections
        skip_connections: list[Tensor] = [x]

        # Encoder path
        for _idx, (encoder_blocks, downsample) in enumerate(
            zip(self.encoder_blocks, self.downsample_blocks, strict=False)
        ):
            for block in encoder_blocks:
                if isinstance(block, AttentionBlock):
                    x = block(x)
                else:
                    x = block(x, t_emb, y_emb)

            skip_connections.append(x)
            x = downsample(x)

        # Bottleneck
        for block in self.mid_blocks:
            if isinstance(block, AttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb, y_emb)

        # if skip_connections and skip_connections[-1].shape[-2:] == x.shape[-2:]:
        #     skip_connections.pop()

        # Decoder path
        for _idx, (decoder_blocks, upsample) in enumerate(
            zip(self.decoder_blocks, self.upsample_blocks, strict=False)
        ):
            # Upsample first (except for first decoder level)
            if not isinstance(upsample, nn.Identity):
                x = upsample(x)

            # Concatenate with skip connection
            if skip_connections:
                skip = skip_connections.pop()
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, x.shape[-2:])
                x = torch.cat([x, skip], dim=1)

            # Process through decoder blocks
            for block in decoder_blocks:
                if isinstance(block, AttentionBlock):
                    x = block(x)
                else:
                    x = block(x, t_emb, y_emb)

        # Output projection
        return self.conv_out(x)

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet().to("cuda")
    console.print(f"Total parameters: {model.count_parameters():,}")
    # console.print(model)
    img_size = (model.img_channels, model.img_size, model.img_size)
    timestep_size = (1,)
    class_size = (1,)
    summary(model, input_size=[img_size, timestep_size, class_size], device="cuda")
