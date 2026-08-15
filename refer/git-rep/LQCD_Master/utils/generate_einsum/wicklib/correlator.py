from typing import Dict, Generator, List, Optional, Tuple, Union

from .operator import Operator
from .tensor import (
    SpinColorTensorType,
    QuarkFieldTensor,
    SpinGammaTensor,
    SpinProjectorTensor,
    ColorDeltaTensor,
    ColorEpsilonTensor,
)
from .adjacency import AdjacencyTerm
from .index import IndexMap


class WickTerm:
    def __init__(
        self,
        factor: Union[int, float, complex],
        tensors: List[Union[QuarkFieldTensor, SpinColorTensorType]],
    ) -> None:
        self.factor = factor
        self.tensors: List[SpinColorTensorType] = []
        self.quark_fields: List[QuarkFieldTensor] = []
        quark_map: Dict[int, int] = {}
        antiquark_map: Dict[int, int] = {}
        self.index_map = IndexMap()
        self.num_quark = 0
        self.num_antiquark = 0
        for tensor in reversed(tensors):
            if isinstance(tensor, QuarkFieldTensor):
                self.quark_fields.append(tensor)
                if not tensor.antiquark:
                    if id(tensor) in quark_map:
                        raise ValueError("quark fields should be unique")
                    quark_map[id(tensor)] = self.num_quark
                    self.index_map.set_spin(tensor.spin, f"α{self.num_quark}")
                    self.index_map.set_color(tensor.color, f"a{self.num_quark}")
                    self.num_quark += 1
                else:
                    if id(tensor) in antiquark_map:
                        raise ValueError("antiquark fields should be unique")
                    antiquark_map[id(tensor)] = self.num_antiquark
                    self.index_map.set_spin(tensor.spin, f"β{self.num_antiquark}")
                    self.index_map.set_color(tensor.color, f"b{self.num_antiquark}")
                    self.num_antiquark += 1
            else:
                self.tensors.append(tensor)
        if self.num_quark != self.num_antiquark:
            raise ValueError("number of quarks and antiquarks should be the same")
        self.tensors = [self._map_tensor_index(tensor) for tensor in self.tensors]

        self.adjacency_terms: List[AdjacencyTerm] = []
        for pairs in self.pair_quark_antiquark([], self.quark_fields):
            adjacency = AdjacencyTerm(self.num_quark, self.factor, self.tensors)
            for sign, quark, antiquark in pairs:
                adjacency.factor *= sign
                row = quark_map[id(quark)]
                col = antiquark_map[id(antiquark)]
                adjacency.matrix[row][col].set(quark, antiquark)
            self.adjacency_terms.append(adjacency)

    def pair_quark_antiquark(
        self, pairs: List[Tuple[int, QuarkFieldTensor, QuarkFieldTensor]], quark_fields: List[QuarkFieldTensor]
    ) -> Generator[List[Tuple[int, QuarkFieldTensor, QuarkFieldTensor]], None, None]:
        if not quark_fields:
            yield list(pairs)
            return
        for quark_idx, quark in enumerate(quark_fields):
            if not quark.antiquark:
                break
        else:
            raise ValueError("unmatched antiquark found")
        found = False
        for antiquark_idx, antiquark in enumerate(quark_fields):
            if antiquark.antiquark and quark.flavor == antiquark.flavor:
                found = True
                distance = quark_idx - antiquark_idx
                distance = -distance if distance < 0 else distance - 1
                pairs.append(((-1) ** distance, quark, antiquark))
                yield from self.pair_quark_antiquark(
                    pairs, [q for q in quark_fields if q is not quark and q is not antiquark]
                )
                pairs.pop()
        if not found:
            raise ValueError("unmatched quark found")

    def _map_tensor_index(self, tensor):
        if isinstance(tensor, SpinGammaTensor):
            return SpinGammaTensor(
                tensor.gamma,
                self.index_map.getdefault_spin(tensor.left),
                self.index_map.getdefault_spin(tensor.right),
            )
        if isinstance(tensor, SpinProjectorTensor):
            return SpinProjectorTensor(
                tensor.factors,
                self.index_map.getdefault_spin(tensor.left),
                self.index_map.getdefault_spin(tensor.right),
            )
        if isinstance(tensor, ColorDeltaTensor):
            return ColorDeltaTensor(
                self.index_map.getdefault_color(tensor.left),
                self.index_map.getdefault_color(tensor.right),
            )
        if isinstance(tensor, ColorEpsilonTensor):
            return ColorEpsilonTensor(
                self.index_map.getdefault_color(tensor.left),
                self.index_map.getdefault_color(tensor.middle),
                self.index_map.getdefault_color(tensor.right),
            )
        return tensor


class Correlator:
    def __init__(self, operator: Operator) -> None:
        self.operator = operator
        self.terms: List[AdjacencyTerm] = []
        for term in self.operator.terms:
            factor, tensors = term.to_tensor()
            wick_term = WickTerm(factor, tensors)
            self.terms.extend(wick_term.adjacency_terms)

    def simplify(self, degenerate: bool = False, order: Optional[List[str]] = None):
        combined = {}
        for term in self.terms:
            term.simplify(degenerate, order)
            key = term.signature()
            if key in combined:
                combined[key].factor += term.factor
            else:
                combined[key] = term
        self.terms = [term for term in combined.values() if term.factor != 0]

    def __str__(self):
        if not self.terms:
            return "0"
        parts = []
        for i, term in enumerate(self.terms):
            text = str(term)
            if i == 0:
                parts.append(text)
            elif text.startswith("-"):
                parts.append(f"- {text[1:]}")
            else:
                parts.append(f"+ {text}")
        return " ".join(parts)
