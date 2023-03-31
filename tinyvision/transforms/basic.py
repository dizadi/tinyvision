from tinygrad import Tensor
from tinyvision.transforms import Transform
from typing import Tuple


class Resize(Transform):
    def __init__(self, resized_shape: Tuple[int, int]) -> None:
        self._shape = resized_shape

    def __call__(self, image: Tensor) -> Tensor:
        """Resizes image to the configured shape."""
        # TODO
        return image
    
class Crop(Transform):
    def __init__(self, top_left_corner, bottom_right_corner) -> None:
        self._xmax, self._ymax = top_left_corner
        self._xmin, self._ymin = bottom_right_corner
    
    def __call__(self, image: Tensor) -> Tensor:
        return image


class Rotate(Transform):
    def __init__(self, rotation_degrees: float) -> None:
        self._rotation_degrees = rotation_degrees

    def __call__(self, image: Tensor) -> Tensor:
        return image