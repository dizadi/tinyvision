"""Utility functions and classes used throughout tinyvision."""
from typing import Type
from PIL import Image
from numpy.typing import NDArray
from tinygrad import Tensor


class TinyImage(Tensor):

    @classmethod
    def from_pil(cls, image: Image) -> Tensor:
        pass

    @classmethod
    def from_numpy(cls, image: NDArray) -> Tensor:
        pass

    @classmethod
    def from_path(cls, image_path: str) -> Tensor:
        pil_image = Image.open(image_path)
        return cls.from_pil(pil_image)

    def to_pil(self) -> Image:
        pass

    def to_numpy(self) -> NDArray:
        pass