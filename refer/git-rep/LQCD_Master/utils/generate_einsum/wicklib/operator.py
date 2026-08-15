from typing import List, Sequence, Tuple, Union

from .gamma import Gamma, C
from .index import ColorIndex, SpinIndex
from .quark import QuarkField
from .tensor import (
    SpinColorTensorType,
    QuarkFieldTensor,
    SpinGammaTensor,
    SpinProjectorTensor,
    ColorDeltaTensor,
    ColorEpsilonTensor,
)


class OperatorTerm:
    def __init__(self, blocks: List["Block"], factor: Union[int, float, complex]) -> None:
        self.blocks = blocks
        self.factor = factor

    def to_tensor(
        self,
    ) -> Tuple[Union[int, float, complex], List[Union[QuarkFieldTensor, SpinColorTensorType]]]:
        result = []
        factor = self.factor
        for block in self.blocks:
            block_factor, tensors = block.to_tensor()
            factor *= block_factor
            result.extend(tensors)
        return factor, result


class Operator:
    def __init__(self, terms: List[OperatorTerm]) -> None:
        self.terms = terms

    def __repr__(self) -> str:
        return f"Operator({len(self.terms)} term(s), factor={[t.factor for t in self.terms]})"

    def adjoint(self) -> "Operator":
        return Operator(
            [
                OperatorTerm([block.adjoint() for block in reversed(term.blocks)], term.factor.conjugate())
                for term in self.terms
            ]
        )

    @classmethod
    def from_block(cls, block: "Block") -> "Operator":
        return cls([OperatorTerm([block], 1)])

    def __mul__(self, rhs: Union[int, float, complex, "Block", "Operator"]) -> "Operator":
        if isinstance(rhs, (int, float, complex)):
            return Operator([OperatorTerm(term.blocks, term.factor * rhs) for term in self.terms])
        elif isinstance(rhs, Block):
            return self * Operator.from_block(rhs)
        elif isinstance(rhs, Operator):
            return Operator(
                [
                    OperatorTerm(lhs_term.blocks + rhs_term.blocks, lhs_term.factor * rhs_term.factor)
                    for lhs_term in self.terms
                    for rhs_term in rhs.terms
                ]
            )
        else:
            return NotImplemented

    def __rmul__(self, lhs: Union[int, float, complex, "Block", "Operator"]) -> "Operator":
        if isinstance(lhs, (int, float, complex)):
            return Operator([OperatorTerm(term.blocks, lhs * term.factor) for term in self.terms])
        elif isinstance(lhs, Block):
            return Operator.from_block(lhs) * self
        elif isinstance(lhs, Operator):
            return Operator(
                [
                    OperatorTerm(lhs_term.blocks + rhs_term.blocks, lhs_term.factor * rhs_term.factor)
                    for lhs_term in lhs.terms
                    for rhs_term in self.terms
                ]
            )
        else:
            return NotImplemented

    def __add__(self, rhs: Union["Block", "Operator"]) -> "Operator":
        if isinstance(rhs, Block):
            return self + Operator.from_block(rhs)
        elif isinstance(rhs, Operator):
            return Operator(self.terms + rhs.terms)
        else:
            return NotImplemented

    def __radd__(self, lhs: Union["Block", "Operator"]) -> "Operator":
        if isinstance(lhs, Block):
            return Operator.from_block(lhs) + self
        elif isinstance(lhs, Operator):
            return Operator(lhs.terms + self.terms)
        else:
            return NotImplemented

    def __neg__(self) -> "Operator":
        return Operator([OperatorTerm(term.blocks, -term.factor) for term in self.terms])

    def __sub__(self, rhs: Union["Block", "Operator"]) -> "Operator":
        if isinstance(rhs, Block):
            return self + (-Operator.from_block(rhs))
        elif isinstance(rhs, Operator):
            return Operator(self.terms + [OperatorTerm(term.blocks, -term.factor) for term in rhs.terms])
        else:
            return NotImplemented

    def __rsub__(self, lhs: Union["Block", "Operator"]) -> "Operator":
        if isinstance(lhs, Block):
            return Operator.from_block(lhs) - self
        elif isinstance(lhs, Operator):
            return Operator([OperatorTerm(term.blocks, -term.factor) for term in self.terms] + lhs.terms)
        else:
            return NotImplemented


class Block:
    def adjoint(self) -> "Block":
        raise NotImplementedError

    def to_tensor(self):
        raise NotImplementedError

    def __mul__(self, rhs: Union[int, float, complex, "Block", Operator]) -> Operator:
        return Operator.from_block(self) * rhs

    def __rmul__(self, lhs: Union[int, float, complex, "Block", Operator]) -> Operator:
        return lhs * Operator.from_block(self)

    def __add__(self, rhs: Union["Block", Operator]) -> Operator:
        return Operator.from_block(self) + rhs

    def __radd__(self, lhs: Union["Block", Operator]) -> Operator:
        return lhs + Operator.from_block(self)

    def __neg__(self) -> Operator:
        return -Operator.from_block(self)

    def __sub__(self, rhs: Union["Block", Operator]) -> Operator:
        return Operator.from_block(self) - rhs

    def __rsub__(self, lhs: Union["Block", Operator]) -> Operator:
        return lhs - Operator.from_block(self)


class SpinProjector(Block):
    def __init__(self, factors: Sequence[Union[int, float, complex]], spin_snk: SpinIndex, spin_src: SpinIndex) -> None:
        if len(factors) != 16:
            raise ValueError("SpinProjector requires 16 factors.")
        self.factors = [factor for factor in factors]
        self.spin_snk = spin_snk
        self.spin_src = spin_src

    def adjoint(self) -> "SpinProjector":
        return SpinProjector(
            [Gamma(i, factor).D.factor for i, factor in enumerate(self.factors)], self.spin_src, self.spin_snk
        )

    def to_tensor(self):
        alpha, beta = self.spin_snk, self.spin_src
        return 1, [SpinProjectorTensor(self.factors, alpha.name, beta.name)]

    @classmethod
    def P_plus(cls, spin_snk: SpinIndex, spin_src: SpinIndex) -> "SpinProjector":
        factors: List[Union[int, float, complex]] = [0 for _ in range(16)]
        factors[0b0000] = 0.5
        factors[0b1000] = 0.5
        return cls(factors, spin_snk, spin_src)

    @classmethod
    def P_minus(cls, spin_snk: SpinIndex, spin_src: SpinIndex) -> "SpinProjector":
        factors: List[Union[int, float, complex]] = [0 for _ in range(16)]
        factors[0b0000] = 0.5
        factors[0b1000] = -0.5
        return cls(factors, spin_snk, spin_src)


class QuarkBlock(Block):
    def __init__(self, q: QuarkField, gamma: Gamma, x: str, spin: SpinIndex, color: ColorIndex) -> None:
        self.q = q
        self.gamma = gamma
        self.x = x
        self.spin = spin
        self.color = color

    def adjoint(self):
        return AntiQuarkBlock(self.q, self.gamma.D, self.x, self.spin, self.color)

    def to_tensor(self):
        alpha, beta = self.spin, SpinIndex.new()
        a, b = self.color, ColorIndex.new()
        return self.gamma.factor, [
            ColorDeltaTensor(a.name, b.name),
            SpinGammaTensor(Gamma(self.gamma.index), alpha.name, beta.name),
            self.q.to_tensor(self.x, False, False, beta, b),
        ]


class AntiQuarkBlock(Block):
    def __init__(self, q: QuarkField, gamma: Gamma, x: str, spin: SpinIndex, color: ColorIndex) -> None:
        self.q = q
        self.gamma = gamma
        self.x = x
        self.spin = spin
        self.color = color

    def adjoint(self):
        return QuarkBlock(self.q, self.gamma.D, self.x, self.spin, self.color)

    def to_tensor(self):
        alpha, beta = self.spin, SpinIndex.new()
        a, b = self.color, ColorIndex.new()
        return self.gamma.factor, [
            ColorDeltaTensor(b.name, a.name),
            self.q.to_tensor(self.x, True, False, beta, b),
            SpinGammaTensor(Gamma(self.gamma.index), beta.name, alpha.name),
        ]


class QuarkBilinearBlock(Block):
    """M = q̄(barred) · Γ · f(unbarred)"""
    def __init__(self, barred: QuarkField, unbarred: QuarkField, gamma: Gamma, x: str) -> None:
        self.barred = barred       # barred antiquark q̄
        self.unbarred = unbarred   # unbarred quark f
        self.gamma = gamma
        self.x = x

    def adjoint(self):
        # M† = f̄ · Γ̄ · q  (swap, Dirac adjoint of gamma)
        return QuarkBilinearBlock(self.unbarred, self.barred, self.gamma.D, self.x)

    def to_tensor(self):
        alpha, beta = SpinIndex.new(), SpinIndex.new()
        a, b = ColorIndex.new(), ColorIndex.new()
        return self.gamma.factor, [
            ColorDeltaTensor(a.name, b.name),
            self.barred.to_tensor(self.x, True, False, alpha, a),
            SpinGammaTensor(Gamma(self.gamma.index), alpha.name, beta.name),
            self.unbarred.to_tensor(self.x, False, False, beta, b),
        ]


class DiquarkBlock(Block):
    def __init__(self, q: QuarkField, f: QuarkField, gamma: Gamma, x: str, color: ColorIndex) -> None:
        self.q = q
        self.f = f
        self.gamma = gamma
        self.x = x
        self.color = color

    def adjoint(self):
        return AntiDiquarkBlock(self.f, self.q, self.gamma.D, self.x, self.color)

    def to_tensor(self):
        alpha, beta = SpinIndex.new(), SpinIndex.new()
        a, b, c = ColorIndex.new(), ColorIndex.new(), self.color
        return self.gamma.factor, [
            ColorEpsilonTensor(a.name, b.name, c.name),
            self.q.to_tensor(self.x, False, True, alpha, a),
            SpinGammaTensor(Gamma(self.gamma.index), alpha.name, beta.name),
            self.f.to_tensor(self.x, False, False, beta, b),
        ]


class AntiDiquarkBlock(Block):
    def __init__(self, q: QuarkField, f: QuarkField, gamma: Gamma, x: str, color: ColorIndex) -> None:
        self.q = q
        self.f = f
        self.gamma = gamma
        self.x = x
        self.color = color

    def adjoint(self):
        return DiquarkBlock(self.f, self.q, self.gamma.D, self.x, self.color)

    def to_tensor(self):
        alpha, beta = SpinIndex.new(), SpinIndex.new()
        a, b, c = ColorIndex.new(), ColorIndex.new(), self.color
        return -self.gamma.factor, [
            ColorEpsilonTensor(a.name, b.name, c.name),
            self.q.to_tensor(self.x, True, False, alpha, a),
            SpinGammaTensor(Gamma(self.gamma.index), alpha.name, beta.name),
            self.f.to_tensor(self.x, True, True, beta, b),
        ]


class Quark:
    def __init__(self, q: QuarkField, gamma: Gamma):
        self.q = q
        self.gamma = gamma

    def adjoint(self):
        return AntiQuark(self.q, self.gamma.D)

    def at(self, x: str, spin: SpinIndex, color: ColorIndex) -> QuarkBlock:
        return QuarkBlock(self.q, self.gamma, x, spin, color)


class AntiQuark:
    def __init__(self, q: QuarkField, gamma: Gamma):
        self.q = q
        self.gamma = gamma

    def adjoint(self):
        return Quark(self.q, self.gamma.D)

    def at(self, x: str, spin: SpinIndex, color: ColorIndex) -> AntiQuarkBlock:
        return AntiQuarkBlock(self.q, self.gamma, x, spin, color)


class QuarkBilinear:
    """M = q̄(barred) · Γ · f(unbarred)"""
    def __init__(self, barred: QuarkField, unbarred: QuarkField, gamma: Gamma):
        self.barred = barred       # barred antiquark q̄
        self.unbarred = unbarred   # unbarred quark f
        self.gamma = gamma

    def adjoint(self):
        return QuarkBilinear(self.unbarred, self.barred, self.gamma.D)

    def at(self, x: str) -> QuarkBilinearBlock:
        return QuarkBilinearBlock(self.barred, self.unbarred, self.gamma, x)


class Diquark:
    def __init__(self, q: QuarkField, f: QuarkField, gamma: Gamma):
        self.q = q
        self.f = f
        self.gamma = gamma

    def adjoint(self):
        return AntiDiquark(self.f, self.q, self.gamma.D)

    def at(self, x: str, color: ColorIndex) -> DiquarkBlock:
        return DiquarkBlock(self.q, self.f, self.gamma, x, color)


class AntiDiquark:
    def __init__(self, q: QuarkField, f: QuarkField, gamma: Gamma):
        self.q = q
        self.f = f
        self.gamma = gamma

    def adjoint(self):
        return Diquark(self.f, self.q, self.gamma.D)

    def at(self, x: str, color: ColorIndex) -> AntiDiquarkBlock:
        return AntiDiquarkBlock(self.q, self.f, self.gamma, x, color)
