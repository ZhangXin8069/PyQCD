from typing import Union


class Gamma:
    popcnt = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]
    popsign = [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1]

    def __init__(self, index: int, factor: Union[int, float, complex] = 1) -> None:
        assert isinstance(index, int) and 0b0000 <= index <= 0b1111, "index should be int from 0 to 15"
        self.index = index
        self.factor = factor

    def __repr__(self) -> str:
        factor = "-" if self.factor == -1 else "" if self.factor == 1 else f"{self.factor} * "
        return (
            f"{factor}"
            f"{'γ₀' if self.index == 0 else ''}"
            f"{'γ₁' if self.index & 0b0001 else ''}"
            f"{'γ₂' if self.index & 0b0010 else ''}"
            f"{'γ₃' if self.index & 0b0100 else ''}"
            f"{'γ₄' if self.index & 0b1000 else ''}"
        )

    def __neg__(self) -> "Gamma":
        return Gamma(self.index, -self.factor)

    def __mul__(self, rhs: Union[int, float, complex]) -> "Gamma":
        if not isinstance(rhs, (int, float, complex)):
            return NotImplemented
        return Gamma(self.index, self.factor * rhs)

    def __rmul__(self, lhs: Union[int, float, complex]) -> "Gamma":
        if not isinstance(lhs, (int, float, complex)):
            return NotImplemented
        return Gamma(self.index, lhs * self.factor)

    def __truediv__(self, rhs: Union[int, float, complex]) -> "Gamma":
        if not isinstance(rhs, (int, float, complex)):
            return NotImplemented
        return Gamma(self.index, self.factor / rhs)

    def __matmul__(self, rhs: "Gamma") -> "Gamma":
        if not isinstance(rhs, Gamma):
            return NotImplemented
        index = self.index ^ rhs.index
        factor = self.factor * rhs.factor
        if self.index & 0b1000:
            factor *= Gamma.popsign[rhs.index & 0b0111]
        if self.index & 0b0100:
            factor *= Gamma.popsign[rhs.index & 0b0011]
        if self.index & 0b0010:
            factor *= Gamma.popsign[rhs.index & 0b0001]
        # if self.index & 0b0001:
        #     factor *= Gamma.popsign[rhs.index & 0b0000]
        return Gamma(index, factor)

    @property
    def T(self) -> "Gamma":
        """Transpose"""
        factor = self.factor
        factor *= -1 if Gamma.popcnt[self.index] & 0b10 else 1
        factor *= Gamma.popsign[self.index & 0b0101]
        return Gamma(self.index, factor)

    @property
    def H(self) -> "Gamma":
        """Hermitian conjugate"""
        factor = self.factor.conjugate()
        factor *= -1 if Gamma.popcnt[self.index] & 0b10 else 1
        return Gamma(self.index, factor)

    @property
    def D(self) -> "Gamma":
        """Dirac conjugate"""
        factor = self.factor.conjugate()
        factor *= -1 if Gamma.popcnt[self.index] & 0b10 else 1
        factor *= Gamma.popsign[self.index & 0b0111]
        return Gamma(self.index, factor)


GAMMA_0 = Gamma(0)  # γ0
GAMMA_1 = Gamma(1)  # γ1
GAMMA_2 = Gamma(2)  # γ2
GAMMA_3 = Gamma(4)  # γ3
GAMMA_4 = Gamma(8)  # γ4
GAMMA_5 = Gamma(15)  # γ5
C = Gamma(10)  # C
