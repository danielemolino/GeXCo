"""Checkpoint-compatible encode, ray-lift, and per-voxel fusion pipeline."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from gexco.reconstruction.encoder import PretrainedViewEncoder, ViTEncoder
from gexco.reconstruction.fusion import AverageViewFusion, WeightedViewFusion3D
from gexco.reconstruction.geometry import ViewMetadata
from gexco.reconstruction.lifting import lift_view_features
from gexco.reconstruction.modulation import FiLMModulation3D


class GeometryConditioner(nn.Module):
    """Construct the 128-channel spatial condition used by the MAISI U-Net.

    Attribute names follow the serialized reconstruction checkpoint exactly:
    ``encoder``, ``weighted_fusion``, and ``modulation``.
    """

    def __init__(
        self,
        encoder: PretrainedViewEncoder | ViTEncoder | None = None,
        *,
        encoder_channels: int = 128,
        fusion_channels: int = 128,
        latent_channels: int = 4,
        use_adaptive_fusion: bool = True,
    ) -> None:
        """Initialize the complete conditioning path.

        Args:
            encoder: Posed pretrained adapter or a lightweight ViT for tests.
            encoder_channels: Encoder feature width.
            fusion_channels: Output conditioning width.
            latent_channels: MAISI latent width.
            use_adaptive_fusion: Use learned fusion instead of averaging.
        """
        super().__init__()
        self.encoder = encoder or PretrainedViewEncoder(
            use_enc_projector=True,
            trainable=True,
        )
        inferred_channels = int(getattr(self.encoder, "embed_dim", encoder_channels))
        self.average_fusion = AverageViewFusion()
        self.weighted_fusion: WeightedViewFusion3D | None = (
            WeightedViewFusion3D(inferred_channels, fusion_channels) if use_adaptive_fusion else None
        )
        self.modulation = FiLMModulation3D(fusion_channels, latent_channels)
        self.fusion_channels = int(fusion_channels if self.weighted_fusion is not None else inferred_channels)

    @staticmethod
    def _as_image_list(
        images: Sequence[torch.Tensor] | torch.Tensor,
    ) -> list[torch.Tensor]:
        """Normalize sequence and stacked-tensor inputs."""
        if isinstance(images, torch.Tensor):
            if images.ndim != 5:
                raise ValueError("Stacked images must have shape (B,V,C,H,W).")
            return list(images.unbind(dim=1))
        return list(images)

    def _encode_view(self, image: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
        """Encode a single posed view into a patch-grid feature map."""
        if hasattr(self.encoder, "encode_tokens"):
            tokens = self.encoder.encode_tokens(image, angle_rad)
        else:
            tokens = self.encoder(image)
        return self.encoder.tokens_to_feature_map(tokens)

    def encode_views(
        self,
        images: Sequence[torch.Tensor] | torch.Tensor,
        views: Sequence[ViewMetadata],
    ) -> list[torch.Tensor]:
        """Encode all input views with their absolute projection angle.

        Args:
            images: View sequence or tensor shaped ``(B,V,1,H,W)``.
            views: Matching acquisition metadata.

        Returns:
            List of feature maps shaped ``(B,C,Hf,Wf)``.
        """
        image_list = self._as_image_list(images)
        if not image_list or len(image_list) != len(views):
            raise ValueError("images and views must contain the same non-zero number of entries.")
        if hasattr(self.encoder, "encode_tokens"):
            stacked = torch.stack(image_list, dim=1)
            batch, count = stacked.shape[:2]
            theta = stacked.new_tensor([math.radians(float(view.angle_deg)) for view in views]).expand(batch, -1)
            tokens = self.encoder.encode_tokens(stacked.flatten(0, 1), theta.reshape(-1))
            maps = self.encoder.tokens_to_feature_map(tokens)
            return list(maps.reshape(batch, count, *maps.shape[1:]).unbind(1))
        return [
            self._encode_view(image, image.new_full((image.shape[0],), math.radians(float(view.angle_deg))))
            for image, view in zip(image_list, views, strict=True)
        ]

    def fuse_views(
        self,
        lifted_views: Sequence[torch.Tensor],
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Fuse lifted volumes.

        Args:
            lifted_views: Sequence of ``(B,C,D,H,W)`` volumes.
            return_weights: Return learned weights when adaptive fusion is active.

        Returns:
            Fused condition or ``(condition, weights)``.
        """
        if self.weighted_fusion is None:
            fused = self.average_fusion(lifted_views)
            if return_weights:
                count = len(lifted_views)
                weights = torch.full(
                    (
                        fused.shape[0],
                        count,
                        1,
                        *fused.shape[-3:],
                    ),
                    1.0 / count,
                    device=fused.device,
                    dtype=fused.dtype,
                )
                return fused, weights
            return fused
        return self.weighted_fusion(lifted_views, return_weights=return_weights)

    def forward(
        self,
        images: Sequence[torch.Tensor] | torch.Tensor,
        views: Sequence[ViewMetadata],
        latent_grid: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | list[torch.Tensor]]:
        """Encode, lift, and fuse one or more views.

        Args:
            images: View sequence or ``(B,V,1,H,W)`` tensor.
            views: Matching view metadata.
            latent_grid: World-coordinate grid shaped ``(B,D,H,W,3)``.
            return_details: Return intermediate features and fusion weights.

        Returns:
            Fused condition shaped ``(B,C,D,H,W)`` or an intermediate dictionary.
        """
        encoded = self.encode_views(images, views)
        lifted = [
            lift_view_features(feature_map, latent_grid, view) for feature_map, view in zip(encoded, views, strict=True)
        ]
        fused, weights = self.fuse_views(lifted, return_weights=True)
        if not return_details:
            return fused
        return {
            "encoded_views": encoded,
            "lifted_views": lifted,
            "fused_volume": fused,
            "latent_conditioning": fused,
            "fusion_weights": weights,
        }


__all__ = ["GeometryConditioner"]
