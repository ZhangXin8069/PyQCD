from typing import List, NamedTuple, Union

from .gamma import Gamma
from .quark import QuarkPrefixType


class QuarkFieldTensor(NamedTuple):
    flavor: str
    prefix: QuarkPrefixType
    location: str
    antiquark: bool
    transpose: bool
    spin: str
    color: str

    def __str__(self):
        if self.antiquark:
            if self.transpose:
                return f"{self.flavor}\u0304^T_{{{self.spin},{self.color}}}({self.location}){self.prefix}"
            else:
                return f"{self.flavor}\u0304_{{{self.spin},{self.color}}}({self.location}){self.prefix}"
        else:
            if self.transpose:
                return f"{self.prefix}{self.flavor}^T_{{{self.spin},{self.color}}}({self.location})"
            else:
                return f"{self.prefix}{self.flavor}_{{{self.spin},{self.color}}}({self.location})"


class SpinGammaTensor(NamedTuple):
    gamma: Gamma
    left: str
    right: str

    def __str__(self):
        return f"({self.gamma})_{{{self.left},{self.right}}}"


class SpinProjectorTensor(NamedTuple):
    factors: List[Union[int, float, complex]]
    left: str
    right: str

    def __str__(self):
        return f"P_{{{self.left},{self.right}}}"


class ColorDeltaTensor(NamedTuple):
    left: str
    right: str

    def __str__(self):
        return f"δ_{{{self.left},{self.right}}}"


class ColorEpsilonTensor(NamedTuple):
    left: str
    middle: str
    right: str

    def __str__(self):
        return f"ϵ_{{{self.left},{self.middle},{self.right}}}"


SpinColorTensorType = Union[SpinGammaTensor, SpinProjectorTensor, ColorDeltaTensor, ColorEpsilonTensor]
