"""Correlated phenomenological Pauli noise: independent channels on a set of
supports Gamma, combined by convolution (Eq. (12)), plus exact moments and
syndrome sampling."""

from __future__ import annotations

import numpy as np

from .pauli import IDENTITY, Pauli, bichar, mul, paulis_on
from .code import StabilizerCode


class LocalChannel:
    """An independent Pauli channel P_gamma supported on a set of qubits."""

    def __init__(self, qubits: tuple[int, ...], probs: dict[Pauli, float]):
        self.qubits = tuple(qubits)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-12, "probabilities must sum to 1"
        self.probs = dict(probs)
        self.paulis = list(probs.keys())
        self.p = np.array([probs[a] for a in self.paulis])

    @classmethod
    def random(cls, qubits: tuple[int, ...], error_rate: float, rng) -> "LocalChannel":
        """Random asymmetric channel with P(I) = 1 - error_rate > 1/2
        (condition (2) of Definition 1)."""
        assert error_rate < 0.5
        errors = list(paulis_on(qubits))
        w = rng.random(len(errors))
        w = error_rate * w / w.sum()
        probs = {IDENTITY: 1.0 - error_rate}
        for e, pe in zip(errors, w):
            probs[e] = pe
        return cls(qubits, probs)

    def moment(self, a: Pauli) -> float:
        """E_gamma(a) = sum_e <a, e> P_gamma(e)  (Fourier transform, Eq. (6))."""
        return sum(bichar(a, e) * pe for e, pe in self.probs.items())


class CorrelatedNoise:
    """P = *_{gamma in Gamma} P_gamma  (Eq. (12))."""

    def __init__(self, channels: list[LocalChannel]):
        self.channels = channels
        self.supports = [c.qubits for c in channels]

    # -- exact quantities ------------------------------------------------
    def moment(self, a: Pauli) -> float:
        """E(a) = prod_gamma E_gamma(a): Fourier turns convolution into products (Eq. (9))."""
        m = 1.0
        for c in self.channels:
            m *= c.moment(a)
        return m

    def full_distribution(self) -> dict[Pauli, float]:
        """Brute-force convolution of all local channels (small n only)."""
        dist: dict[Pauli, float] = {IDENTITY: 1.0}
        for c in self.channels:
            new: dict[Pauli, float] = {}
            for e, pe in dist.items():
                for f, pf in c.probs.items():
                    g = mul(e, f)
                    new[g] = new.get(g, 0.0) + pe * pf
            dist = new
        return dist

    def exact_syndrome_distribution(self, code: StabilizerCode) -> np.ndarray:
        q = np.zeros(2 ** (code.n - code.k))
        for e, pe in self.full_distribution().items():
            q[code.syndrome(e)] += pe
        return q

    # -- sampling ---------------------------------------------------------
    def sample_syndromes(self, code: StabilizerCode, shots: int, rng) -> np.ndarray:
        """Simulate `shots` rounds of phenomenological QEC and return the
        empirical syndrome distribution (histogram, normalized)."""
        ex = np.zeros(shots, dtype=np.int64)
        ez = np.zeros(shots, dtype=np.int64)
        for c in self.channels:
            idx = rng.choice(len(c.paulis), size=shots, p=c.p)
            xs = np.array([a[0] for a in c.paulis], dtype=np.int64)
            zs = np.array([a[1] for a in c.paulis], dtype=np.int64)
            ex ^= xs[idx]
            ez ^= zs[idx]
        # syndrome bit i = symplectic form of (g_i, e)
        pop = np.array([bin(v).count("1") for v in range(1 << code.n)], dtype=np.int64)
        sig = np.zeros(shots, dtype=np.int64)
        for i, g in enumerate(code.generators):
            bit = (pop[g[0] & ez] + pop[g[1] & ex]) & 1
            sig |= bit << i
        hist = np.bincount(sig, minlength=2 ** (code.n - code.k))
        return hist / shots


def wht(q: np.ndarray) -> np.ndarray:
    """Walsh-Hadamard transform: syndrome distribution -> stabilizer moments.

    E(s_c) = sum_sigma (-1)^{|c & sigma|} q(sigma), matching the ordering of
    StabilizerCode.stabilizers.  This is exactly Eq. (6) restricted to S:
    stabilizer moments are measurable from syndrome statistics alone.
    """
    a = q.copy().astype(float)
    h = 1
    while h < len(a):
        for i in range(0, len(a), 2 * h):
            x = a[i : i + h].copy()
            y = a[i + h : i + 2 * h].copy()
            a[i : i + h] = x + y
            a[i + h : i + 2 * h] = x - y
        h *= 2
    return a
