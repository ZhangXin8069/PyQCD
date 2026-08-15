from typing import Literal, Optional, Union

from .index import ColorIndex, SpinIndex


class QuarkLocal:
    def __init__(self) -> None:
        pass

    def adjoint(self):
        return self

    def __str__(self):
        return ""


class QuarkShift:
    def __init__(self, location: str) -> None:
        self.location = location

    def adjoint(self):
        return self

    def __str__(self):
        return ""


class QuarkLink:
    def __init__(self, location: str, origin: Optional[str] = None) -> None:
        self.location = location
        self.origin = origin

    def set_origin(self, origin: str) -> "QuarkLink":
        return QuarkLink(self.location, origin)

    def adjoint(self):
        if self.origin is None:
            raise ValueError("cannot take adjoint of a shift without origin")
        return QuarkLink(self.origin, self.location)

    def __str__(self):
        return f"W({self.origin},{self.location})"


class QuarkSmearing:
    def __init__(self, rho: str) -> None:
        self.rho = rho

    def adjoint(self):
        return self

    def __str__(self):
        return f"(φ_{self.rho})"


class QuarkDerivative:
    def __init__(self, direction: str) -> None:
        self.direction = direction

    def adjoint(self):
        return self

    def __str__(self):
        return f"(∇_{self.direction})"


QuarkPrefixType = Union[QuarkLocal, QuarkShift, QuarkLink, QuarkSmearing, QuarkDerivative]
QuarkFlavorType = Literal["u", "d", "s", "c", "b", "t"]


class QuarkField:
    def __init__(self, flavor: QuarkFlavorType, prefix: QuarkPrefixType = QuarkLocal()) -> None:
        self.flavor = flavor
        self.prefix = prefix

    def to_tensor(self, origin: str, antiquark: bool, transpose: bool, spin: SpinIndex, color: ColorIndex):
        from .tensor import QuarkFieldTensor

        prefix = self.prefix
        if isinstance(prefix, QuarkLink):
            prefix = prefix.set_origin(origin)
        if antiquark:
            prefix = prefix.adjoint()
        location = origin
        if isinstance(self.prefix, (QuarkShift, QuarkLink)):
            location = self.prefix.location
        return QuarkFieldTensor(self.flavor, prefix, location, antiquark, transpose, spin.name, color.name)

    @classmethod
    def shift(cls, flavor: QuarkFlavorType, location: str) -> "QuarkField":
        return cls(flavor, QuarkShift(location))

    @classmethod
    def link(cls, flavor: QuarkFlavorType, location: str) -> "QuarkField":
        return cls(flavor, QuarkLink(location))

    @classmethod
    def smearing(cls, flavor: QuarkFlavorType, rho: str) -> "QuarkField":
        return cls(flavor, QuarkSmearing(rho))

    @classmethod
    def derivative(cls, flavor: QuarkFlavorType, location: str) -> "QuarkField":
        return cls(flavor, QuarkDerivative(location))
