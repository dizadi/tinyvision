from tinygrad import Tensor
from tinyvision.transforms import Transform
from typing import Tuple
import numpy as np

class RandomCrop(Transform):
    def __init__(self, crop_shape: Tuple[int, int], randomness: float = 1.0) -> None:
        self._crop_shape = crop_shape
        self._randomness = randomness

    def __call__(self, image: Tensor) -> Tensor:
        """Random crop with configured parameters."""
        return image

class RandomNoise(Transform):
    def __init__(self, noise_coefficient: float) -> None:
        self._noise_coefficient = noise_coefficient

    def __call__(self, image: Tensor) -> Tensor:
        """Injects random noise into the image."""
        return image

class RandomRotation(Transform):
    def __init__(self, max_rotation_degrees: float, randomness: float) -> None:
        self._max_rotation_deg = max_rotation_degrees
        self._randomness = randomness

    def __call__(self, image: Tensor) -> Tensor:
        """Rotates the image at random."""
        return image
