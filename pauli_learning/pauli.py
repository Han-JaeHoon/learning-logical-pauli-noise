"""Effective Pauli group P^n = P^n / {±1, ±i} in binary symplectic representation.

A Pauli operator is a pair of integers (x, z) interpreted as bitmasks over n qubits:
qubit i carries X iff bit i of x is set, Z iff bit i of z is set (Y = both).
Phases are dropped, exactly as in the paper (effective Pauli group).
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, Iterator, Tuple

Pauli = Tuple[int, int]  # (x bitmask, z bitmask)

IDENTITY: Pauli = (0, 0)

_CHARS = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}


def popcount(v: int) -> int:
    return bin(v).count("1")


def mul(a: Pauli, b: Pauli) -> Pauli:
    """Group multiplication in the effective Pauli group (phases dropped)."""
    return (a[0] ^ b[0], a[1] ^ b[1])


def bichar(a: Pauli, e: Pauli) -> int:
    """Bicharacter <a, e> of Eq. (1): +1 if a and e commute, -1 otherwise."""
    return 1 - 2 * ((popcount(a[0] & e[1]) + popcount(a[1] & e[0])) % 2)


def commutes(a: Pauli, b: Pauli) -> bool:
    return bichar(a, b) == 1


def support(a: Pauli) -> int:
    """Bitmask of qubits on which a acts nontrivially."""
    return a[0] | a[1]


def weight(a: Pauli) -> int:
    return popcount(support(a))


def is_substring(b: Pauli, a: Pauli) -> bool:
    """b <= a in the sense of the paper: a restricted to supp(b) equals b."""
    sup = support(b)
    return ((b[0] ^ a[0]) & sup) == 0 and ((b[1] ^ a[1]) & sup) == 0


def substrings(a: Pauli) -> Iterator[Pauli]:
    """All b <= a (including identity and a itself)."""
    qubits = [i for i in range(support(a).bit_length()) if (support(a) >> i) & 1]
    for keep in product([0, 1], repeat=len(qubits)):
        x, z = 0, 0
        for k, q in zip(keep, qubits):
            if k:
                x |= a[0] & (1 << q)
                z |= a[1] & (1 << q)
        yield (x, z)


def paulis_on(qubits: Iterable[int], include_identity: bool = False) -> Iterator[Pauli]:
    """All Paulis supported inside the given set of qubits."""
    qubits = list(qubits)
    for letters in product("IXYZ", repeat=len(qubits)):
        if not include_identity and all(c == "I" for c in letters):
            continue
        x, z = 0, 0
        for c, q in zip(letters, qubits):
            if c in "XY":
                x |= 1 << q
            if c in "ZY":
                z |= 1 << q
        yield (x, z)


def all_paulis(n: int) -> Iterator[Pauli]:
    yield from paulis_on(range(n), include_identity=True)


def to_str(a: Pauli, n: int) -> str:
    return "".join(
        _CHARS[((a[0] >> i) & 1, (a[1] >> i) & 1)] for i in range(n)
    )


def from_str(s: str) -> Pauli:
    x, z = 0, 0
    for i, c in enumerate(s):
        if c in "XY":
            x |= 1 << i
        if c in "ZY":
            z |= 1 << i
        if c not in "IXYZ":
            raise ValueError(f"invalid Pauli character {c!r}")
    return (x, z)
