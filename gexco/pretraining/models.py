"""Checkpoint-compatible GeXCo world-model architecture.

This module reproduces the exact ViT, absolute-pose encoder, AdaLN predictor,
and paired token projectors stored in the released GeXCo pretraining state.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from gexco.reconstruction.encoder import (
    TransformerBlock,
    ViTEncoder,
    sinusoidal_pose_encoding,
)


def _get_1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    """Build one-dimensional sinusoidal positional features."""
    if embed_dim % 2:
        raise ValueError("embed_dim must be even.")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10_000**omega)
    phases = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate([np.sin(phases), np.cos(phases)], axis=1)


def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Build the fixed predictor position encoding.

    Args:
        embed_dim: Token dimension.
        grid_size: Square patch-grid edge.

    Returns:
        Tensor shaped ``(grid_size**2, embed_dim)``.
    """
    if embed_dim % 2:
        raise ValueError("embed_dim must be even.")
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, grid_size, grid_size)
    embedding_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    embedding_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return torch.from_numpy(np.concatenate([embedding_h, embedding_w], axis=1)).float()


def _apply_masks(tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Gather a per-sample subset of tokens."""
    indices = masks.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return torch.gather(tokens, dim=1, index=indices)


class WorldModelEncoder(nn.Module):
    """Shared ViT encoder with learnable absolute-view pose injection."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        input_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 12,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        out_channels: int = 128,
        pose_scale_learnable: bool = True,
        no_encoder_pose: bool = False,
    ) -> None:
        """Initialize the shared encoder.

        Args:
            in_channels: Input image channels.
            input_size: Nominal image resolution.
            patch_size: ViT patch size.
            embed_dim: ViT token dimension.
            depth: ViT depth.
            num_heads: Attention head count.
            mlp_ratio: ViT MLP ratio.
            out_channels: Feature-map output width.
            pose_scale_learnable: Learn a scalar multiplying pose features.
            no_encoder_pose: Disable pose addition for ablations.
        """
        super().__init__()
        if no_encoder_pose and pose_scale_learnable:
            raise ValueError("no_encoder_pose and pose_scale_learnable are mutually exclusive.")
        self.encoder = ViTEncoder(
            in_channels=in_channels,
            input_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_channels=out_channels,
        )
        if pose_scale_learnable:
            self.pose_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        else:
            self.register_parameter("pose_scale", None)
        self.no_encoder_pose = bool(no_encoder_pose)
        self._last_pose_norm_ratio = 0.0

    @property
    def grid_size(self) -> int:
        """Return the nominal patch-grid edge.

        Returns:
            Number of patches along one image axis.
        """
        return self.encoder.grid_size

    @property
    def num_patches(self) -> int:
        """Return the nominal patch count.

        Returns:
            Total number of patch tokens.
        """
        return self.encoder.num_patches

    @property
    def embed_dim(self) -> int:
        """Return the token dimension.

        Returns:
            Encoder embedding width.
        """
        return self.encoder.embed_dim

    def current_pose_scale(self) -> float:
        """Return the effective absolute-pose scale.

        Returns:
            Current scalar multiplying the sinusoidal pose embedding.
        """
        return 1.0 if self.pose_scale is None else float(self.pose_scale.detach())

    def current_pose_norm_ratio(self) -> float:
        """Return the last pose-to-token norm ratio.

        Returns:
            Mean pose-feature norm divided by image-token norm.
        """
        return float(self._last_pose_norm_ratio)

    def forward(
        self,
        images: torch.Tensor,
        theta: torch.Tensor | None = None,
        *,
        return_global: bool = False,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode posed radiographs.

        Args:
            images: Tensor shaped ``(B,C,H,W)``.
            theta: Absolute angles in radians shaped ``(B,)``.
            return_global: Also return mean-pooled embeddings.
            masks: Optional token indices shaped ``(B,M)``.

        Returns:
            Patch tokens, optionally paired with global embeddings.
        """
        if images.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(images.shape)}.")
        batch_size = images.shape[0]
        if theta is None:
            theta = torch.zeros(batch_size, device=images.device, dtype=images.dtype)
        if theta.shape != (batch_size,):
            raise ValueError(f"Expected theta shape {(batch_size,)}, got {tuple(theta.shape)}.")

        patches = self.encoder.patch_embed(images)
        patch_height, patch_width = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = tokens + self.encoder._pos_embed_for_grid(
            patch_height,
            patch_width,
            patches.device,
            patches.dtype,
        )
        if not self.no_encoder_pose:
            pose = sinusoidal_pose_encoding(theta.to(dtype=patches.dtype), self.embed_dim)
            effective_pose = pose.unsqueeze(1)
            if self.pose_scale is not None:
                effective_pose = self.pose_scale.to(dtype=tokens.dtype) * effective_pose
            with torch.no_grad():
                self._last_pose_norm_ratio = float(
                    effective_pose.norm(dim=-1).mean() / (tokens.norm(dim=-1).mean() + 1e-6)
                )
            tokens = tokens + effective_pose
        else:
            self._last_pose_norm_ratio = 0.0
        if masks is not None:
            tokens = _apply_masks(tokens, masks)
        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)
        global_embedding = tokens.mean(dim=1)
        return (tokens, global_embedding) if return_global else tokens

    def forward_feature_map(
        self,
        images: torch.Tensor,
        theta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode posed images into a square 2D feature map.

        Args:
            images: Tensor shaped ``(B,C,H,W)``.
            theta: Absolute angles in radians shaped ``(B,)``.

        Returns:
            Encoded feature map shaped ``(B,Cf,H/P,W/P)``.
        """
        tokens = self(images, theta=theta)
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError("Unexpected encoder output.")
        return self.encoder.tokens_to_feature_map(tokens)


def modulate(tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive LayerNorm affine modulation.

    Args:
        tokens: Normalized token tensor.
        shift: Learned conditional shift.
        scale: Learned conditional scale.

    Returns:
        Affine-modulated token tensor.
    """
    return tokens * (1.0 + scale) + shift


class AdaLNBlock(nn.Module):
    """AdaLN-Zero predictor block used by the trained checkpoint."""

    def __init__(self, dim: int, heads: int, mlp_dim: int) -> None:
        """Initialize a predictor block."""
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Transform tokens conditioned on an angular embedding.

        Args:
            tokens: Patch tokens shaped ``(B,N,C)``.
            condition: Angular features shaped ``(B,C)``.

        Returns:
            Conditioned tokens with the input shape.
        """
        conditioned = condition.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioned).chunk(
            6, dim=-1
        )
        attention_input = modulate(self.norm1(tokens), shift_msa, scale_msa)
        attention_output, _ = self.attn(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )
        tokens = tokens + gate_msa * attention_output
        return tokens + gate_mlp * self.mlp(modulate(self.norm2(tokens), shift_mlp, scale_mlp))


class TokenProjector(nn.Module):
    """Two-layer token projector stored in the world-model checkpoint."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
    ) -> None:
        """Initialize the ``Linear-BN-GELU-Linear`` projector."""
        super().__init__()
        self.output_dim = int(output_dim or input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project tokens while preserving batch and token axes.

        Args:
            tokens: Input shaped ``(B,N,C)``.

        Returns:
            Projected tokens shaped ``(B,N,output_dim)``.
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected (B,N,D), got {tuple(tokens.shape)}.")
        batch_size, num_tokens, _ = tokens.shape
        projected = self.net(tokens.reshape(batch_size * num_tokens, -1))
        return projected.reshape(batch_size, num_tokens, self.output_dim)


class ViewPredictor(nn.Module):
    """Predict target-view tokens from anchor tokens and angular deltas."""

    def __init__(
        self,
        *,
        num_patches: int = 256,
        input_dim: int = 128,
        output_dim: int | None = None,
        predictor_embed_dim: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        init_tau: float = 0.1,
        init_affinity_tau: float = 0.1,
        use_adaln: bool = True,
        use_action_mlp_embed: bool = False,
        action_mlp_ratio: float = 4.0,
    ) -> None:
        """Initialize the checkpoint-compatible predictor."""
        super().__init__()
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError("num_patches must describe a square grid.")
        self.num_patches = int(num_patches)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim or input_dim)
        self.predictor_embed_dim = int(predictor_embed_dim)
        self.use_adaln = bool(use_adaln)
        self.predictor_embed = nn.Linear(self.input_dim, self.predictor_embed_dim, bias=True)
        self.action_embed: nn.Module | None = None
        if use_action_mlp_embed:
            hidden = int(action_mlp_ratio * self.predictor_embed_dim)
            self.action_embed = nn.Sequential(
                nn.Linear(1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.predictor_embed_dim),
            )
        self.register_buffer(
            "predictor_pos_embed",
            build_2d_sincos_pos_embed(self.predictor_embed_dim, grid_size).unsqueeze(0),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(
                    dim=self.predictor_embed_dim,
                    heads=num_heads,
                    mlp_dim=int(self.predictor_embed_dim * mlp_ratio),
                )
                if self.use_adaln
                else TransformerBlock(self.predictor_embed_dim, num_heads, mlp_ratio)
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(self.predictor_embed_dim)
        self.proj = nn.Linear(self.predictor_embed_dim, self.output_dim, bias=True)
        self.tau_log = nn.Parameter(torch.log(torch.tensor(float(init_tau))))
        self.affinity_tau_log = nn.Parameter(torch.log(torch.tensor(float(init_affinity_tau))))

    def _forward_flat(self, tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict one target for each item in a flattened batch."""
        features = self.predictor_embed(tokens)
        features = features + self.predictor_pos_embed.to(
            device=features.device,
            dtype=features.dtype,
        )
        condition = sinusoidal_pose_encoding(
            action.to(device=features.device, dtype=features.dtype),
            self.predictor_embed_dim,
        )
        if self.action_embed is not None:
            condition = condition + self.action_embed(action.to(features.dtype).unsqueeze(1))
        if self.use_adaln:
            for block in self.blocks:
                features = block(features, condition)
            return self.proj(self.norm(features))
        features = torch.cat([condition.unsqueeze(1), features], dim=1)
        for block in self.blocks:
            features = block(features)
        return self.proj(self.norm(features)[:, 1:])

    def forward(self, tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict one or ``K`` target views.

        Args:
            tokens: Anchor tokens shaped ``(B,N,D)``.
            action: Delta angles shaped ``(B,)``, ``(B,1)``, or ``(B,K)``.

        Returns:
            ``(B,N,D)`` for one target or ``(B,K,N,D)`` for multiple targets.
        """
        if action.ndim == 1:
            return self._forward_flat(tokens, action)
        if action.ndim != 2 or action.shape[0] != tokens.shape[0]:
            raise ValueError("action must have shape (B,), (B,1), or (B,K).")
        if action.shape[1] == 1:
            return self._forward_flat(tokens, action[:, 0])
        batch_size, targets = action.shape
        repeated = (
            tokens[:, None]
            .expand(-1, targets, -1, -1)
            .reshape(
                batch_size * targets,
                tokens.shape[1],
                tokens.shape[2],
            )
        )
        predicted = self._forward_flat(repeated, action.reshape(-1))
        return predicted.reshape(
            batch_size,
            targets,
            predicted.shape[1],
            predicted.shape[2],
        )


class WorldModel(nn.Module):
    """Shared encoder, predictor, and paired projection heads."""

    def __init__(
        self,
        encoder: WorldModelEncoder,
        predictor: ViewPredictor,
        enc_projector: TokenProjector | None = None,
        pred_projector: TokenProjector | None = None,
    ) -> None:
        """Initialize the complete pretraining model."""
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.enc_projector = enc_projector
        self.pred_projector = pred_projector

    def encode(self, images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Encode and project posed radiographs.

        Args:
            images: Radiographs shaped ``(B,1,H,W)``.
            theta: Absolute view angles in radians.

        Returns:
            Projected encoder tokens.
        """
        tokens = self.encoder(images, theta=theta)
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError("Unexpected encoder output.")
        return self.enc_projector(tokens) if self.enc_projector is not None else tokens

    def forward(
        self,
        anchor: torch.Tensor,
        angular_delta: torch.Tensor,
        anchor_angles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict target tokens for one or several delta angles.

        Args:
            anchor: Anchor radiographs shaped ``(B,1,H,W)``.
            angular_delta: One or ``K`` relative target angles.
            anchor_angles: Optional absolute anchor angles.

        Returns:
            Predicted and projected target tokens.
        """
        if anchor_angles is None:
            anchor_angles = torch.zeros(
                anchor.shape[0],
                device=anchor.device,
                dtype=anchor.dtype,
            )
        tokens = self.encode(anchor, anchor_angles)
        predicted = self.predictor(tokens, angular_delta)
        if self.pred_projector is None:
            return predicted
        if predicted.ndim == 3:
            return self.pred_projector(predicted)
        batch_size, targets, num_tokens, channels = predicted.shape
        projected = self.pred_projector(predicted.reshape(batch_size * targets, num_tokens, channels))
        return projected.reshape(batch_size, targets, num_tokens, projected.shape[-1])


def build_world_model(
    *,
    input_size: int = 128,
    patch_size: int = 8,
    embed_dim: int = 128,
    depth: int = 12,
    num_heads: int = 4,
    predictor_depth: int = 3,
    predictor_num_heads: int = 4,
    projector_hidden_dim: int = 512,
    projector_dim: int = 128,
    use_projector: bool = True,
    pose_scale_learnable: bool = True,
) -> WorldModel:
    """Build the exact default paper world model.

    Args:
        input_size: Nominal radiograph resolution.
        patch_size: ViT patch edge.
        embed_dim: Encoder and predictor token width.
        depth: Encoder Transformer depth.
        num_heads: Encoder attention-head count.
        predictor_depth: Predictor Transformer depth.
        predictor_num_heads: Predictor attention-head count.
        projector_hidden_dim: Hidden width of both projection heads.
        projector_dim: Output width of both projection heads.
        use_projector: Instantiate the paired heads.
        pose_scale_learnable: Learn the absolute-angle pose scale.

    Returns:
        Complete checkpoint-compatible world model.
    """
    encoder = WorldModelEncoder(
        input_size=input_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        out_channels=embed_dim,
        pose_scale_learnable=pose_scale_learnable,
    )
    effective_dim = projector_dim if use_projector else embed_dim
    predictor = ViewPredictor(
        num_patches=encoder.num_patches,
        input_dim=effective_dim,
        output_dim=effective_dim,
        predictor_embed_dim=embed_dim,
        depth=predictor_depth,
        num_heads=predictor_num_heads,
        use_adaln=True,
    )
    enc_projector = TokenProjector(embed_dim, projector_hidden_dim, projector_dim) if use_projector else None
    pred_projector = TokenProjector(effective_dim, projector_hidden_dim, projector_dim) if use_projector else None
    return WorldModel(encoder, predictor, enc_projector, pred_projector)


def load_world_model_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[WorldModel, dict[str, Any]]:
    """Restore the development checkpoint without key conversion.

    Args:
        checkpoint_path: Path to ``best.pt`` or ``last.pt``.
        map_location: PyTorch load target.

    Returns:
        Restored world model and serialized checkpoint metadata.
    """
    payload = torch.load(
        Path(checkpoint_path),
        map_location=map_location,
        weights_only=False,
    )
    config = payload["config"]
    model = build_world_model(
        input_size=int(config["input_size"]),
        patch_size=int(config["patch_size"]),
        embed_dim=int(config["embed_dim"]),
        depth=int(config["depth"]),
        num_heads=int(config["num_heads"]),
        predictor_depth=int(config["predictor_depth"]),
        predictor_num_heads=int(config["predictor_num_heads"]),
        projector_hidden_dim=int(config.get("proj_hidden_dim", 512)),
        projector_dim=int(config.get("proj_dim", config["embed_dim"])),
        use_projector=bool(config.get("use_projector", False)),
        pose_scale_learnable=bool(config.get("pose_scale_learnable", False)),
    )
    model.encoder.load_state_dict(payload["encoder"], strict=True)
    model.predictor.load_state_dict(payload["view_predictor"], strict=True)
    if model.enc_projector is not None:
        model.enc_projector.load_state_dict(payload["enc_projector"], strict=True)
    if model.pred_projector is not None:
        model.pred_projector.load_state_dict(payload["pred_projector"], strict=True)
    return model, payload


__all__ = [
    "AdaLNBlock",
    "TokenProjector",
    "TransformerBlock",
    "ViewPredictor",
    "WorldModel",
    "WorldModelEncoder",
    "build_2d_sincos_pos_embed",
    "build_world_model",
    "load_world_model_checkpoint",
]
