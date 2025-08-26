"""Feature extraction module for U-Flow model.

This module implements feature extraction functionality for the U-Flow model for
anomaly detection. It provides:

1. Feature extractors based on different backbone architectures
2. Utility function to get appropriate feature extractor
3. Support for multiple scales of feature extraction

Supported backbone architectures:
- ResNet-based: resnet18, wide_resnet50_2
- Vision Transformer: mcait (CaiT models)
- DINO/DINOv2: vit_small_patch16_224.dino, vit_base_patch16_224.dino,
  vit_large_patch14_dinov2.lvd142m

Example:
    >>> from anomalib.models.image.uflow.feature_extraction import get_feature_extractor
    >>> extractor = get_feature_extractor(backbone="resnet18")
    >>> features = extractor(torch.randn(1, 3, 256, 256))
    >>> 
    >>> # Using DINO backbone
    >>> dino_extractor = get_feature_extractor(backbone="vit_small_patch16_224.dino")
    >>> dino_features = dino_extractor(torch.randn(1, 3, 224, 224))

See Also:
    - :func:`get_feature_extractor`: Factory function to get feature extractors
    - :class:`LayerNormFeatureExtractor`: Main feature extractor implementation
    - :class:`CaitFeatureExtractor`: Alternative feature extractor
    - :class:`DINOFeatureExtractor`: DINO/DINOv2 based feature extractor
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

import timm
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from anomalib.models.components.feature_extractors import TimmFeatureExtractor

AVAILABLE_EXTRACTORS = [
    "mcait", 
    "resnet18", 
    "wide_resnet50_2",
    "vit_small_patch16_224.dino",
    "vit_base_patch16_224.dino", 
    "vit_large_patch14_dinov2.lvd142m"
]


def get_feature_extractor(backbone: str, input_size: tuple[int, int] = (256, 256), tap_blocks: bool = False, use_pos_embedding: bool = True) -> nn.Module:
    """Get feature extractor based on specified backbone architecture.

    This function returns a feature extractor model based on the specified backbone
    architecture. Currently supported backbones are defined in ``AVAILABLE_EXTRACTORS``.

    Args:
        backbone (str): Name of the backbone architecture to use. Must be one of
            ``["mcait", "resnet18", "wide_resnet50_2", "vit_small_patch16_224.dino", 
            "vit_base_patch16_224.dino", "vit_large_patch14_dinov2.lvd142m"]``.
        input_size (tuple[int, int], optional): Input image dimensions as
            ``(height, width)``. Defaults to ``(256, 256)``. Note that DINO models
            have fixed input sizes (224 or 518) regardless of this parameter.

    Returns:
        nn.Module: Feature extractor model instance.

    Raises:
        ValueError: If ``backbone`` is not one of the supported architectures in
            ``AVAILABLE_EXTRACTORS``.

    Example:
        >>> from anomalib.models.image.uflow.feature_extraction import get_feature_extractor
        >>> extractor = get_feature_extractor(backbone="resnet18")
        >>> features = extractor(torch.randn(1, 3, 256, 256))
        >>> 
        >>> # Using DINO backbone
        >>> dino_extractor = get_feature_extractor(backbone="vit_small_patch16_224.dino")
        >>> dino_features = dino_extractor(torch.randn(1, 3, 224, 224))

    See Also:
        - :class:`LayerNormFeatureExtractor`: Main feature extractor implementation
        - :class:`CaitFeatureExtractor`: Alternative feature extractor
        - :class:`DINOFeatureExtractor`: DINO/DINOv2 based feature extractor
    """
    if backbone not in AVAILABLE_EXTRACTORS:
        msg = f"Feature extractor must be one of {AVAILABLE_EXTRACTORS}."
        raise ValueError(msg)

    feature_extractor: nn.Module
    if backbone in {"resnet18", "wide_resnet50_2"}:
        feature_extractor = LayerNormFeatureExtractor(
            backbone,
            input_size,
            layers=("layer1", "layer2", "layer3"),
        ).eval()
    elif backbone == "mcait":
        feature_extractor = CaitFeatureExtractor().eval()
    elif backbone in {"vit_small_patch16_224.dino", "vit_base_patch16_224.dino", 
                      "vit_large_patch14_dinov2.lvd142m"}:
        feature_extractor = DINOFeatureExtractor(backbone, tap_blocks=tap_blocks, use_pos_embed=use_pos_embedding).eval()

    return feature_extractor


class LayerNormFeatureExtractor(TimmFeatureExtractor):
    """Feature extractor based on ResNet (or others) backbones.

    This class extends TimmFeatureExtractor to extract and normalize features from
    common CNN backbones like ResNet. It adds layer normalization to the extracted
    features.

    Args:
        backbone (str): Name of the backbone CNN architecture to use for feature
            extraction (e.g. ``"resnet18"``, ``"wide_resnet50_2"``).
        input_size (tuple[int, int]): Input image dimensions as ``(height, width)``
            used for computing normalization layers.
        layers (tuple[str, ...], optional): Names of layers from which to extract
            features. Defaults to ``("layer1", "layer2", "layer3")``.
        **kwargs: Additional keyword arguments (unused).

    Example:
        >>> import torch
        >>> extractor = LayerNormFeatureExtractor(
        ...     backbone="resnet18",
        ...     input_size=(256, 256)
        ... )
        >>> features = extractor(torch.randn(1, 3, 256, 256))

    Attributes:
        channels (list[int]): Number of channels in each extracted feature layer.
        scale_factors (list[int]): Downsampling factor for each feature layer.
        scales (range): Range object for iterating over feature scales.
        feature_normalizations (nn.ModuleList): Layer normalization modules for
            each feature scale.
    """

    def __init__(
        self,
        backbone: str,
        input_size: tuple[int, int],
        layers: tuple[str, ...] = ("layer1", "layer2", "layer3"),
        **kwargs,  # noqa: ARG002 | unused argument
    ) -> None:
        super().__init__(backbone, layers, pre_trained=True, requires_grad=False)
        self.channels = self.feature_extractor.feature_info.channels()
        self.scale_factors = self.feature_extractor.feature_info.reduction()
        self.scales = range(len(self.scale_factors))

        self.feature_normalizations = nn.ModuleList()
        for in_channels, scale in zip(self.channels, self.scale_factors, strict=True):
            self.feature_normalizations.append(
                nn.LayerNorm(
                    [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                    elementwise_affine=True,
                ),
            )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Extract and normalize features from input image.

        Args:
            img (torch.Tensor): Input image tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            torch.Tensor: Normalized features from multiple network layers.
        """
        features = self.extract_features(img)
        return self.normalize_features(features)

    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """Extract features from input image using backbone network.

        Args:
            img (torch.Tensor): Input image tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            torch.Tensor: Features extracted from multiple network layers.
        """
        self.feature_extractor.eval()
        return self.feature_extractor(img)

    def normalize_features(self, features: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        """Apply layer normalization to extracted features.

        Args:
            features (Iterable[torch.Tensor]): Features extracted from multiple
                network layers.

        Returns:
            list[torch.Tensor]: Normalized features from each layer.
        """
        return [self.feature_normalizations[i](feature) for i, feature in enumerate(features)]


class CaitFeatureExtractor(nn.Module):
    """Feature extractor based on MCait backbone.

    This class implements the feature extractor proposed in the U-Flow paper. It uses two
    independently trained CaiT models at different scales:
    - A CaiT-M48 model with input size 448x448
    - A CaiT-S24 model with input size 224x224

    Each model extracts features at a different scale, and includes normalization layers.

    Example:
        >>> from anomalib.models.image.uflow.feature_extraction import CaitFeatureExtractor
        >>> extractor = CaitFeatureExtractor()
        >>> image = torch.randn(1, 3, 448, 448)
        >>> features = extractor(image)
        >>> [f.shape for f in features]
        [torch.Size([1, 768, 28, 28]), torch.Size([1, 384, 14, 14])]

    Attributes:
        input_size (int): Size of input images (448)
        extractor1 (nn.Module): CaiT-M48 model for scale 1 (448x448)
        extractor2 (nn.Module): CaiT-S24 model for scale 2 (224x224)
        channels (list[int]): Number of channels for each scale [768, 384]
        scale_factors (list[int]): Downsampling factors for each scale [16, 32]
        scales (range): Range object for iterating over scales
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_size = 448
        self.extractor1 = timm.create_model("cait_m48_448", pretrained=True)
        self.extractor2 = timm.create_model("cait_s24_224", pretrained=True)
        self.channels = [768, 384]
        self.scale_factors = [16, 32]
        self.scales = range(len(self.scale_factors))

        for param in self.extractor1.parameters():
            param.requires_grad = False
        for param in self.extractor2.parameters():
            param.requires_grad = False

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Extract and normalize features from input image.

        Args:
            img (torch.Tensor): Input image tensor of shape
                ``(batch_size, channels, height, width)``

        Returns:
            torch.Tensor: List of normalized feature tensors from each scale
        """
        features = self.extract_features(img)
        return self.normalize_features(features)

    def extract_features(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features from input image using both CaiT models.

        The features are extracted at two scales:
        - Scale 1: Using CaiT-M48 up to block index 40 (448x448 input)
        - Scale 2: Using CaiT-S24 up to block index 20 (224x224 input)

        Args:
            img (torch.Tensor): Input image tensor of shape
                ``(batch_size, channels, height, width)``

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Features from both extractors with shapes:
                ``[(B, 768, H/16, W/16), (B, 384, H/32, W/32)]``
        """
        self.extractor1.eval()
        self.extractor2.eval()

        # Scale 1 --> Extractor 1
        x1 = self.extractor1.patch_embed(img)
        x1 = x1 + self.extractor1.pos_embed
        x1 = self.extractor1.pos_drop(x1)
        for i in range(41):  # paper Table 6. Block Index = 40
            x1 = self.extractor1.blocks[i](x1)

        # Scale 2 --> Extractor 2
        img_sub = F.interpolate(torch.Tensor(img), size=(224, 224), mode="bicubic", align_corners=True)
        x2 = self.extractor2.patch_embed(img_sub)
        x2 = x2 + self.extractor2.pos_embed
        x2 = self.extractor2.pos_drop(x2)
        for i in range(21):
            x2 = self.extractor2.blocks[i](x2)

        return (x1, x2)

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize extracted features from both scales.

        For each scale:
        1. Apply layer normalization
        2. Reshape features to spatial format
        3. Append to list of normalized features

        Args:
            features (torch.Tensor): Tuple of features from both extractors

        Returns:
            torch.Tensor: List of normalized feature tensors with shapes:
                ``[(B, 768, H/16, W/16), (B, 384, H/32, W/32)]``
        """
        normalized_features = []
        for i, extractor in enumerate([self.extractor1, self.extractor2]):
            batch, _, channels = features[i].shape
            scale_factor = self.scale_factors[i]

            x = extractor.norm(features[i].contiguous())
            x = x.permute(0, 2, 1)
            x = x.reshape(batch, channels, self.input_size // scale_factor, self.input_size // scale_factor)
            normalized_features.append(x)

        return normalized_features


class DINOFeatureExtractor(nn.Module):
    """Feature extractor based on DINO and DINOv2 Vision Transformer backbones.

    This extractor aligns with U-Flow expectations: multi-scale, normalized outputs.
    Optionally taps intermediate transformer blocks for additional feature levels.
    Optionally disables positional embeddings to avoid position bias.

    Args:
        backbone (str): Name of the DINO/DINOv2 model.
        tap_blocks (bool): If True, taps an intermediate block for a second feature scale.
        use_pos_embed (bool): If False, disables adding positional embedding to patch tokens.
    """

    def __init__(self, backbone: str, tap_blocks: bool = False, use_pos_embed: bool = True) -> None:
        super().__init__()

        if backbone in {"vit_small_patch16_224.dino", "vit_base_patch16_224.dino"}:
            self.input_size = 224
        elif backbone == "vit_large_patch14_dinov2.lvd142m":
            self.input_size = 518
        else:
            raise ValueError(f"Unsupported DINO backbone: {backbone}")

        self.tap_blocks = tap_blocks
        self.use_pos_embed = use_pos_embed
        print(f"self.use_pos_embed: {self.use_pos_embed}")

        self.model = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,
        )
        self.model.eval()

        # Extract model-specific parameters
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_embed.patch_size[0]  # assumes square patches

        # Define output scales (main + optionally intermediate)
        self.channels = [self.embed_dim]
        self.scale_factors = [self.patch_size]

        if tap_blocks:
            self.channels.append(self.embed_dim)
            self.scale_factors.append(self.patch_size * 2)  # Simulated lower-resolution level

        self.scales = range(len(self.scale_factors))

        # Normalizations per scale
        self.feature_normalizations = nn.ModuleList()
        for scale in self.scale_factors:
            out_dim = self.input_size // scale
            self.feature_normalizations.append(
                nn.LayerNorm(
                    [self.embed_dim, out_dim, out_dim],
                    elementwise_affine=True,
                )
            )

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass to extract normalized features from DINO model."""
        if img.shape[-1] != self.input_size:
            img = F.interpolate(img, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)

        features = self.extract_features(img)
        return self.normalize_features(features)

    def extract_features(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Extracts patch tokens from ViT encoder.

        Returns:
            list[Tensor]: [final_block_features, optional_intermediate_block_features]
        """
        x = self.model.patch_embed(img)  # (B, N, C)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # === THIS IS THE ONLY CHANGE ===
        if self.use_pos_embed:
            x = x + self.model.pos_embed
        # ==============================

        x = self.model.pos_drop(x)

        tokens = []
        if self.tap_blocks:
            tap_idx = len(self.model.blocks) // 2  # Halfway block
            for i, blk in enumerate(self.model.blocks):
                x = blk(x)
                if i == tap_idx:
                    tokens.append(x.clone())
        else:
            for blk in self.model.blocks:
                x = blk(x)

        tokens.append(x)
        return tokens

    def normalize_features(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Converts patch tokens to spatial maps and applies normalization."""
        normed_features = []

        for i, feature in enumerate(features):
            # Remove CLS token and reshape
            x = feature[:, 1:, :]  # (B, N, C)
            batch_size, num_patches, channels = x.shape
            h = w = int(num_patches ** 0.5)

            x = x.permute(0, 2, 1).reshape(batch_size, channels, h, w)
            x = self.feature_normalizations[i](x)
            normed_features.append(x)

        return normed_features

