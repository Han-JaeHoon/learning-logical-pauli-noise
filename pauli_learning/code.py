"""Stabilizer codes: stabilizer group S, logical operators L = S^perp,
syndromes, and correctable regions (paper, "Stabilizer codes" section)."""

from __future__ import annotations

from .pauli import (
    IDENTITY,
    Pauli,
    all_paulis,
    bichar,
    commutes,
    from_str,
    mul,
    popcount,
    support,
    to_str,
    weight,
)


class StabilizerCode:
    """A stabilizer code given by a list of independent, commuting generators.

    Attributes
    ----------
    stabilizers : list[Pauli]
        All 2^(n-k) elements of S, ordered so that stabilizers[c] = prod_{i in c} g_i
        where c is read as a bitmask over generators.  This ordering matches the
        Walsh-Hadamard transform of the syndrome distribution.
    logicals : list[Pauli]
        All 2^(n+k) elements of L = S^perp (the annihilator, Eq. (2)).
    """

    def __init__(self, generators: list[str] | list[Pauli], n: int | None = None):
        if generators and isinstance(generators[0], str):
            self.n = len(generators[0])
            self.generators = [from_str(g) for g in generators]
        else:
            assert n is not None
            self.n = n
            self.generators = list(generators)
        m = len(self.generators)
        for i in range(m):
            for j in range(i + 1, m):
                assert commutes(self.generators[i], self.generators[j]), (
                    f"generators {i}, {j} do not commute"
                )
        self.k = self.n - m

        # Enumerate S; index c encodes which generators are multiplied.
        self.stabilizers: list[Pauli] = []
        for c in range(2 ** m):
            s = IDENTITY
            for i in range(m):
                if (c >> i) & 1:
                    s = mul(s, self.generators[i])
            self.stabilizers.append(s)
        assert len(set(self.stabilizers)) == 2 ** m, "generators are not independent"
        self._stab_set = set(self.stabilizers)

        # L = S^perp: everything commuting with all generators (Eq. (2)).
        self.logicals: list[Pauli] = [
            a for a in all_paulis(self.n)
            if all(commutes(a, g) for g in self.generators)
        ]
        assert len(self.logicals) == 2 ** (self.n + self.k)

    # ------------------------------------------------------------------
    def syndrome(self, e: Pauli) -> int:
        """Syndrome S(e) as a bitmask: bit i = 1 iff <g_i, e> = -1."""
        sig = 0
        for i, g in enumerate(self.generators):
            if bichar(g, e) == -1:
                sig |= 1 << i
        return sig

    def is_stabilizer(self, a: Pauli) -> bool:
        return a in self._stab_set

    def is_logical(self, a: Pauli) -> bool:
        return all(commutes(a, g) for g in self.generators)

    # ------------------------------------------------------------------
    def is_correctable_region(self, region: int) -> bool:
        """A region R (qubit bitmask) is correctable iff it only supports
        trivial logical operators (paper, discussion before Definition 1)."""
        for l in self.logicals:
            if support(l) & ~region == 0 and l not in self._stab_set:
                return False
        return True

    def distance(self) -> int:
        return min(weight(l) for l in self.logicals if l not in self._stab_set)

    def pure_distance(self) -> int:
        """Minimum weight of a nontrivial element of L (stabilizers included) --
        the quantity limiting the *physical* channel estimation of Ref. [26]."""
        return min(weight(l) for l in self.logicals if l != IDENTITY)

    # ------------------------------------------------------------------
    def coset_representatives(self):
        """Representatives of the cosets P^n / S, labelled by
        (syndrome, logical class).  Logical classes are labelled by coset
        representatives of L / S with minimal weight (e.g. I, X_L, Y_L, Z_L)."""
        # logical classes: cosets of S inside L
        seen: set[Pauli] = set()
        logical_reps: list[Pauli] = []
        for l in sorted(self.logicals, key=weight):
            if l in seen:
                continue
            logical_reps.append(l)
            for s in self.stabilizers:
                seen.add(mul(l, s))
        # pure errors: one representative per syndrome (minimum weight)
        pure: dict[int, Pauli] = {}
        for e in sorted(all_paulis(self.n), key=weight):
            sig = self.syndrome(e)
            if sig not in pure:
                pure[sig] = e
            if len(pure) == 2 ** (self.n - self.k):
                break
        return pure, logical_reps

    def __repr__(self) -> str:
        gens = ", ".join(to_str(g, self.n) for g in self.generators)
        return f"StabilizerCode[[{self.n},{self.k},{self.distance()}]]({gens})"


# ----------------------------------------------------------------------
FIVE_QUBIT = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]

SHOR = [
    "ZZIIIIIII", "IZZIIIIII",   # Z-type, block 1
    "IIIZZIIII", "IIIIZZIII",   # Z-type, block 2
    "IIIIIIZZI", "IIIIIIIZZ",   # Z-type, block 3
    "XXXXXXIII", "IIIXXXXXX",   # X-type, between blocks
]
