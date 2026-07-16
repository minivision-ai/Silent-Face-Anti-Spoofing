import numpy as np
import torch
from PIL import Image


class ToTensor:
    """Convert a PIL Image or numpy.ndarray (H x W x C) to a (C x H x W) float tensor.

    Unlike torchvision's ToTensor, values are left in their original range
    (not divided by 255) to match how the shipped model weights were trained.
    """

    def __call__(self, pic):
        if isinstance(pic, Image.Image):
            pic = np.array(pic)
        if pic.ndim == 2:
            pic = pic[:, :, None]
        return torch.from_numpy(pic.transpose((2, 0, 1))).float()
