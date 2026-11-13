"""Checkpoint-compatible ViT encoders for GeXCo radiograph conditioning.

The module mirrors the encoder used by the released world-model checkpoint:
an image ViT is wrapped by an absolute-view pose embedding and, optionally,
the learned ``128 -> 512 -> 128`` token projector used during pretraining.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_pose_encoding(theta: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Encode absolute or relative projection angles.

    Args:
        theta: Angles in radians with shape ``(B,)``.
        embed_dim: Output feature dimension.

    Returns:
        Sinusoidal features with shape ``(B, embed_dim)``.
    """
    if theta.ndim != 1:
        raise ValueError(f"Expected theta with shape (B,), got {tuple(theta.shape)}.")
    index = torch.arange(embed_dim, device=theta.device, dtype=theta.dtype)
    phase = (torch.remainder(index, 2) * (math.pi / 2.0)).abs()
    frequency = torch.floor(index / 2.0) + 1.0
    return torch.sin(frequency.unsqueeze(0) * theta.unsqueeze(1) + phase.unsqueeze(0))


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer block used by the paper ViT."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        """Initialize the block.

        Args:
            embed_dim: Token dimension.
            num_heads: Self-attention head count.
            mlp_ratio: MLP expansion ratio.
        """
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Transform tokens while preserving their shape.

        Args:
            tokens: Tensor shaped ``(B,N,C)``.

        Returns:
            Transformed tokens shaped ``(B,N,C)``.
        """
        normalized = self.norm1(tokens)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class ViTEncoder(nn.Module):
    """Patch-based image ViT with state names matching the trained encoder."""

    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 12,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        out_channels: int | None = None,
    ) -> None:
        """Initialize the ViT.

        Args:
            in_channels: Radiograph channels.
            input_size: Nominal square input resolution.
            patch_size: Square patch edge.
            embed_dim: Transformer token dimension.
            depth: Transformer depth.
            num_heads: Attention head count.
            mlp_ratio: MLP expansion ratio.
            out_channels: Optional feature-map output width.
        """
        super().__init__()
        if input_size <= 0 or patch_size <= 0 or input_size % patch_size:
            raise ValueError("input_size must be positive and divisible by patch_size.")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.in_channels = int(in_channels)
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.out_channels = int(out_channels or embed_dim)
        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size**2
        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(self.embed_dim, num_heads, mlp_ratio) for _ in range(int(depth))])
        self.norm = nn.LayerNorm(self.embed_dim)
        self.proj: nn.Module = (
            nn.Identity()
            if self.embed_dim == self.out_channels
            else nn.Conv2d(self.embed_dim, self.out_channels, kernel_size=1, bias=False)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos_embed_for_grid(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Interpolate positional embeddings for a non-default patch grid."""
        position = self.pos_embed.to(device=device, dtype=dtype)
        if (height, width) == (self.grid_size, self.grid_size):
            return position
        position_2d = position.reshape(1, self.grid_size, self.grid_size, self.embed_dim).permute(0, 3, 1, 2)
        position_2d = F.interpolate(
            position_2d,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        return position_2d.permute(0, 2, 3, 1).reshape(1, height * width, self.embed_dim)

    def forward(self, images: torch.Tensor, theta: torch.Tensor | None = None) -> torch.Tensor:
        """Encode radiographs into patch tokens.

        ``theta`` is accepted for API compatibility; absolute-view injection is
        performed by :class:`gexco.pretraining.models.WorldModelEncoder`.

        Args:
            images: Tensor shaped ``(B,C,H,W)``.
            theta: Optional unused angle tensor.

        Returns:
            Patch tokens shaped ``(B,N,embed_dim)``.
        """
        del theta
        if images.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(images.shape)}.")
        patches = self.patch_embed(images)
        height, width = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = tokens + self._pos_embed_for_grid(height, width, patches.device, patches.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def tokens_to_feature_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape square patch tokens into a 2D feature map.

        Args:
            tokens: Tensor shaped ``(B,N,C)`` with square ``N``.

        Returns:
            Feature map shaped ``(B,out_channels,sqrt(N),sqrt(N))``.
        """
        grid_size = int(round(tokens.shape[1] ** 0.5))
        if grid_size * grid_size != tokens.shape[1]:
            raise ValueError("Token count must be a perfect square.")
        features = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_size, grid_size)
        return self.proj(features)

    def forward_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images and return the patch-grid feature map.

        Args:
            images: Tensor shaped ``(B,C,H,W)``.

        Returns:
            Feature map shaped ``(B,out_channels,H/P,W/P)``.
        """
        return self.tokens_to_feature_map(self(images))


class PretrainedViewEncoder(nn.Module):
    """World-model encoder adapter used by the reconstruction checkpoint.

    The adapter exposes the pretrained image encoder and token projector under
    GeXCo-specific, descriptive module names.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        input_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 12,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        in_channels: int = 1,
        pose_scale_learnable: bool = True,
        use_enc_projector: bool = True,
        projector_hidden_dim: int = 512,
        projector_dim: int = 128,
        trainable: bool = True,
        require_reconstruction_architecture: bool = False,
    ) -> None:
        """Initialize or restore the view encoder.

        Args:
            checkpoint_path: Pretraining checkpoint with separate component states.
            input_size: Radiograph resolution when no checkpoint is supplied.
            patch_size: ViT patch size.
            embed_dim: ViT token dimension.
            depth: ViT depth.
            num_heads: ViT attention heads.
            mlp_ratio: ViT MLP ratio.
            in_channels: Radiograph channels.
            pose_scale_learnable: Learn the absolute-view embedding scale.
            use_enc_projector: Apply the pretrained encoder projection head.
            projector_hidden_dim: Projection-head hidden width.
            projector_dim: Projection-head output width.
            trainable: Whether encoder and projector receive gradients.
            require_reconstruction_architecture: Require the paper encoder when loading pretraining weights.
        """
        super().__init__()
        from gexco.pretraining.models import TokenProjector, WorldModelEncoder

        payload: dict[str, Any] | None = None
        config: dict[str, Any] = {}
        if checkpoint_path:
            payload = torch.load(
                Path(checkpoint_path),
                map_location="cpu",
                weights_only=False,
            )
            config = dict(payload.get("config") or {})
            if not config:
                raise RuntimeError("World-model checkpoint does not contain `config`.")
            if require_reconstruction_architecture:
                from gexco.reconstruction.configuration import validate_world_model_initialization

                validate_world_model_initialization(config)

        self.view_encoder = WorldModelEncoder(
            in_channels=int(config.get("in_channels", in_channels)),
            input_size=int(config.get("input_size", input_size)),
            patch_size=int(config.get("patch_size", patch_size)),
            embed_dim=int(config.get("embed_dim", embed_dim)),
            depth=int(config.get("depth", depth)),
            num_heads=int(config.get("num_heads", num_heads)),
            mlp_ratio=float(config.get("mlp_ratio", mlp_ratio)),
            out_channels=int(config.get("embed_dim", embed_dim)),
            pose_scale_learnable=bool(config.get("pose_scale_learnable", pose_scale_learnable)),
            no_encoder_pose=bool(config.get("no_encoder_pose", False)),
        )
        if payload is not None:
            state = payload.get("encoder")
            if state is None:
                raise KeyError("World-model checkpoint does not contain `encoder`.")
            self.view_encoder.load_state_dict(state, strict=True)

        use_checkpoint_projector = bool(config.get("use_projector", use_enc_projector))
        self.enc_projector: TokenProjector | None = None
        effective_dim = self.view_encoder.embed_dim
        if use_enc_projector and use_checkpoint_projector:
            self.enc_projector = TokenProjector(
                input_dim=effective_dim,
                hidden_dim=int(config.get("proj_hidden_dim", projector_hidden_dim)),
                output_dim=int(config.get("proj_dim", projector_dim)),
            )
            if payload is not None:
                projector_state = payload.get("enc_projector")
                if projector_state is None:
                    raise KeyError("World-model checkpoint does not contain `enc_projector`.")
                self.enc_projector.load_state_dict(projector_state, strict=True)
            effective_dim = int(config.get("proj_dim", projector_dim))

        self.embed_dim = effective_dim
        self.grid_size = self.view_encoder.grid_size
        self.num_patches = self.view_encoder.num_patches
        self.trainable = bool(trainable)
        for parameter in self.parameters():
            parameter.requires_grad_(self.trainable)
        if not self.trainable:
            self.eval()

    def train(self, mode: bool = True) -> "PretrainedViewEncoder":
        """Set training mode while keeping a frozen encoder in evaluation mode.

        Args:
            mode: Requested training state.

        Returns:
            This encoder instance.
        """
        super().train(mode and self.trainable)
        return self

    def encode_tokens(self, images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Encode images with absolute pose and the encoder projector.

        Args:
            images: Tensor shaped ``(B,1,H,W)``.
            theta: Absolute view angles in radians shaped ``(B,)``.

        Returns:
            Projected patch tokens shaped ``(B,N,128)``.
        """
        context = nullcontext() if self.trainable else torch.no_grad()
        with context:
            tokens = self.view_encoder(images, theta=theta)
            if self.enc_projector is not None:
                tokens = self.enc_projector(tokens)
        return tokens

    def forward(self, images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Encode a batch of posed radiographs.

        Args:
            images: Tensor shaped ``(B,1,H,W)``.
            theta: Absolute view angles in radians.

        Returns:
            Projected patch tokens.
        """
        return self.encode_tokens(images, theta)

    def tokens_to_feature_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape projected tokens to their square feature grid.

        Args:
            tokens: Tensor shaped ``(B,N,C)``.

        Returns:
            Feature map shaped ``(B,C,sqrt(N),sqrt(N))``.
        """
        grid_size = int(round(tokens.shape[1] ** 0.5))
        if grid_size * grid_size != tokens.shape[1]:
            raise ValueError("Token count must be a perfect square.")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_size, grid_size)


__all__ = [
    "PretrainedViewEncoder",
    "TransformerBlock",
    "ViTEncoder",
    "sinusoidal_pose_encoding",
]
