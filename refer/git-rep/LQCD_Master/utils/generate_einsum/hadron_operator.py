"""hadron_operator — Hadron operator = list of Tensors (no γ matrix algebra system).

Each operator is a list of Tensors. Individual Tensors carry type, name, flavor, and
spin/color index labels. The contraction engine matches indices by label.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# ── Tensor types ──

@dataclass
class Tensor:
    type: str          # 'gamma', 'epsilon', 'delta', 'quark', 'antiquark', 'projector'
    name: str          # Variable name: 'G5','g1','epsilon','prop_l' etc.
    indices: Tuple[str, ...] = ()   # Index labels: uppercase=spin, lowercase=color
    flavor: str = ''   # Flavor: 'u','d','s','c'

    def __repr__(self):
        return f"{self.name}{self.indices}" + (f"({self.flavor})" if self.flavor else "")


# ── Operator = list of Tensors ──

@dataclass
class Operator:
    tensors: List[Tensor] = field(default_factory=list)
    def __mul__(self, other: "Operator") -> "Operator":
        return Operator(self.tensors + other.tensors)


# ── Convenience γ matrix tensor constructors ──

def gamma(name: str, left: str, right: str) -> Tensor:
    return Tensor("gamma", {"g5": "G5", "g1": "g1", "gx": "g1",
                            "gy": "g2", "gz": "g3", "gt": "g4",
                            "gtg5": "gtg5",
                            "g2": "g2", "g3": "g3", "g4": "g4"}.get(name, name),
                  (left, right))

def epsilon(a: str, b: str, c: str) -> Tensor:
    return Tensor("epsilon", "epsilon", (a, b, c))

def delta(i: str, j: str) -> Tensor:
    return Tensor("delta", "", (i, j))

def quark(flavor: str, spin: str, color: str, location: str = "") -> Tensor:
    return Tensor("quark", "q", (spin, color), flavor)

def antiquark(flavor: str, spin: str, color: str, location: str = "") -> Tensor:
    return Tensor("antiquark", "qb", (spin, color), flavor)

def projector(name: str, left: str, right: str) -> Tensor:
    return Tensor("projector", name, (left, right))


# ── Standard hadron operators ──

def meson_operator(anti_quark_flavor: str, quark_flavor: str, gamma_mat: str,
                    location: str = "") -> Operator:
    """M(x) = q̄(anti_quark_flavor) · Γ · f(quark_flavor)

    Parameters:
        anti_quark_flavor: flavor of q̄ (barred quark)
        quark_flavor: flavor of f (unbarred quark)
        gamma_mat: γ matrix name
    """
    α, β = "A", "B"; a, b = "a", "b"
    is_baryon = False;
    return Operator([
        antiquark(anti_quark_flavor, α, a),
        gamma(gamma_mat, α, β),
        quark(quark_flavor, β, b),
        delta(a, b),
    ]), is_baryon





def baryon_operator(a_flavor: str, b_flavor: str, c_flavor: str,
                    diquark_gamma: str = "Cg5",
                    location: str = "") -> Operator:
    """B(x) = ε_{abc} (q_a^T · (diquark_gamma) · q_b) · q_c

    The full diquark gamma is, e.g.:
      diquark_gamma="Cg5" → Cγ₅ (Jᴾ=1/2⁺ octet, flavor-antisymmetric diquark)
      diquark_gamma="Cg1" → Cγ₁ (Jᴾ=3/2⁺ decuplet, flavor-symmetric diquark)
    """
    α1, β1, γ1 = "A", "B", "C"
    a1, b1, c1 = "a", "b", "c"
    is_baryon = True
    return Operator([
        epsilon(a1, b1, c1),
        quark(a_flavor, α1, a1),
        gamma(diquark_gamma, α1, β1),   # C @ γ_mat connects diquark
        quark(b_flavor, β1, b1),
        quark(c_flavor, γ1, c1),   # spectator (no γ₀)
    ]),is_baryon




def current_operator(flavor_out: str, flavor_in: str, gamma_mat: str,
                     location: str = "") -> Operator:
    """J(x) = q̄(flavor_out) · Γ · q(flavor_in)

    The flavor-changing current operator for three-point functions.

    Parameters
    ----------
    flavor_in : str     Flavor of the quark being annihilated (e.g. "s").
    flavor_out : str    Flavor of the anti-quark being created (e.g. "u").
    gamma_mat : str     Dirac matrix (e.g. "g1" for vector current).
    """
    α, β = "A", "B"; a, b = "a", "b"
    is_baryon = False
    return Operator([
        antiquark(flavor_out, α, a),
        gamma(gamma_mat, α, β),
        quark(flavor_in, β, b),
        delta(a, b),
    ]), is_baryon



