"""Dataset for performing inference on images.

This module provides a dataset class for loading and preprocessing images for
inference in anomaly detection tasks.

Example:
    >>> from pathlib import Path
    >>> from anomalib.data import PredictDataset
    >>> dataset = PredictDataset(path="path/to/images")
    >>> item = dataset[0]
    >>> item.image.shape  # doctest: +SKIP
    torch.Size([3, 256, 256])
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

from torch.utils.data.dataset import Dataset
from torchvision.transforms.v2 import Transform

from anomalib.data import ImageBatch, ImageItem
from anomalib.data.utils import get_image_filenames, read_image, convert_image
import torch


class PredictDataset(Dataset):
    def __init__(
        self,
        images: list[torch.Tensor] | None = None,
        path: str | Path | None = None,
        transform: Transform | None = None,
        image_size: int | tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()

        if images is not None and path is not None:
            raise ValueError("Provide either 'images' or 'path', not both.")

        if images is not None:
            self.images = images
            self.image_filenames = [f"image_{i}.png" for i in range(len(images))]  # Fake filenames
        elif path is not None:
            self.image_filenames = get_image_filenames(path)
            self.images = None
        else:
            raise ValueError("Either 'images' or 'path' must be provided.")

        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> ImageItem:
        if self.images is not None:
            unprocessed_image = self.images[index]
            image = convert_image(unprocessed_image, as_tensor=True)
            image_path = self.image_filenames[index]  # Fake name to keep consistency
        else:
            image_filename = self.image_filenames[index]
            image = read_image(image_filename, as_tensor=True)
            image_path = str(image_filename)

        if self.transform:
            image = self.transform(image)

        return ImageItem(image=image, image_path=image_path)

    @property
    def collate_fn(self) -> Callable:
        return ImageBatch.collate
    
