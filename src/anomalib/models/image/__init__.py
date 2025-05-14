"""Anomalib Image Models.

This module contains implementations of various deep learning models for image-based
anomaly detection.

Example:
    >>> from anomalib.models.image import Padim, Patchcore
    >>> from anomalib.data import MVTecAD  # doctest: +SKIP
    >>> from anomalib.engine import Engine  # doctest: +SKIP

    >>> # Initialize model and data
    >>> datamodule = MVTecAD()  # doctest: +SKIP
    >>> model = Padim()  # doctest: +SKIP
    >>> # Train using the Engine

    >>> engine = Engine()  # doctest: +SKIP
    >>> engine.fit(model=model, datamodule=datamodule)  # doctest: +SKIP

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)  # doctest: +SKIP

Available Models:
    - :class:`Cfa`: Contrastive Feature Aggregation
    - :class:`Cflow`: Conditional Normalizing Flow
    - :class:`Csflow`: Conditional Split Flow
    - :class:`Dfkde`: Deep Feature Kernel Density Estimation
    - :class:`Dfm`: Deep Feature Modeling
    - :class:`Draem`: Dual Reconstruction by Adversarial Masking
    - :class:`Dsr`: Deep Spatial Reconstruction
    - :class:`EfficientAd`: Efficient Anomaly Detection
    - :class:`Fastflow`: Fast Flow
    - :class:`Fre`: Feature Reconstruction Error
    - :class:`Ganomaly`: Generative Adversarial Networks
    - :class:`Padim`: Patch Distribution Modeling
    - :class:`Patchcore`: Patch Core
    - :class:`ReverseDistillation`: Reverse Knowledge Distillation
    - :class:`Stfpm`: Student-Teacher Feature Pyramid Matching
    - :class:`SuperSimpleNet`: SuperSimpleNet
    - :class:`Uflow`: Unsupervised Flow
    - :class:`VlmAd`: Vision Language Model Anomaly Detection
    - :class:`WinClip`: Zero-/Few-Shot CLIP-based Detection
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .cfa import Cfa
from .patchcore import Patchcore
from .uflow import Uflow

__all__ = [
    "Cfa",
    "Cflow",
    "Csflow",
    "Dfkde",
    "Dfm",
    "Draem",
    "Dsr",
    "EfficientAd",
    "Fastflow",
    "Fre",
    "Ganomaly",
    "Padim",
    "Patchcore",
    "ReverseDistillation",
    "Stfpm",
    "Supersimplenet",
    "Uflow",
    "VlmAd",
    "WinClip",
]
