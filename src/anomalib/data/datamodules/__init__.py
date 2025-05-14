"""Anomalib Data Modules."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .depth import Folder3D, MVTec3D
from .image import BTech, Datumaro, Folder, Kolektor, MVTec, MVTecAD, Visa
from .video import Avenue, ShanghaiTech, UCSDped

__all__ = [
    "Folder3D",
    "MVTec3D",
    "BTech",
    "Datumaro",
    "Folder",
    "Kolektor",
    "MVTec",  # Include MVTec for backward compatibility
    "MVTecAD",
    "Visa",
    "Avenue",
    "ShanghaiTech",
    "UCSDped",
]
