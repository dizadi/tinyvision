from typing import List, Optional, Tuple, Any, Callable, Union
from numpy.typing import NDArray
import abc
from PIL import Image
from tinygrad import Tensor

ImageType = Union[Tensor, Image.Image, NDArray]

class Transform(Callable, abc.ABC):
    """Base transform object. Defines interfaces."""
    
    @abc.abstractmethod
    def __call__(self, image: ImageType) -> ImageType:
        """Transform call."""

class Compose(Transform):
    """Wrapper to run multiple transforms in a sequence."""
    def __init__(
        self,
        transforms_list: List[Transform]
    ) -> None:
        """Stores the transforms list"""
        self._transforms_list = transforms_list

    def __call__(self, image: ImageType) -> ImageType:
        """Runs a sequence of transforms."""
        for transform in self._transforms_list:
            image = transform(image)
        return image


