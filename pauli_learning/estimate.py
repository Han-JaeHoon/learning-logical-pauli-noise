"""Estimation of the logical channel from syndrome statistics.

Implements the identifiability machinery of the paper:
  - Gamma' (Eq. (15)): canonical-moment labels inside the noise supports
  - coefficient matrices D_S, D_L (Eq. (18))
  - the rank condition (Eq. (19)) and its verification
  - the log-linear estimator (Eq. (17)) : log E(s) = sum_{a <= s, a in Gamma'} log F(a)
  - reconstruction of the logical channel P_L from logical moments (Eqs. (5)/(11))
"""

from __future__ import annotations

import numpy as np

from .pauli import IDENTITY, Pauli, bichar, is_substring, mul, paulis_on, support
from .code import StabilizerCode


def gamma_prime(supports: list[tuple[int, ...]]) -> list[Pauli]:
    """Gamma' = {a != I : supp(a) subseteq gamma for some gamma in Gamma} (Eq. (15))."""
    out: set[Pauli] = set()
    for qubits in supports:
        out.update(paulis_on(qubits))
    return sorted(out)


def coefficient_matrix(operators: list[Pauli], gamma_p: list[Pauli]) -> np.ndarray:
    """D[s, a] = 1 iff a <= s  (Eq. (18)); rows = operators, columns = Gamma'."""
    D = np.zeros((len(operators), len(gamma_p)))
    for i, s in enumerate(operators):
        for j, a in enumerate(gamma_p):
            if is_substring(a, s):
                D[i, j] = 1.0
    return D


class LogicalEstimator:
    """Estimate logical moments E_L (hence the logical channel) from
    stabilizer moments E_S measured via syndrome statistics."""

    def __init__(self, code: StabilizerCode, supports: list[tuple[int, ...]]):
        self.code = code
        self.gamma_p = gamma_prime(supports)
        self.D_S = coefficient_matrix(code.stabilizers, self.gamma_p)
        self.D_L = coefficient_matrix(code.logicals, self.gamma_p)

    # -- identifiability -------------------------------------------------
    def rank_report(self) -> dict:
        r_S = np.linalg.matrix_rank(self.D_S)
        r_SL = np.linalg.matrix_rank(np.vstack([self.D_S, self.D_L]))
        return {
            "n_params": len(self.gamma_p),          # canonical moments = free params
            "rank_D_S": int(r_S),
            "rank_D_S_and_D_L": int(r_SL),
            "logical_identifiable": bool(r_S == r_SL),   # Eq. (19)
            "physical_identifiable": bool(r_S == len(self.gamma_p)),
        }

    def check_definition_1(self, supports: list[tuple[int, ...]]) -> tuple[bool, list]:
        """Condition (1) of Definition 1: union of any two supports is correctable."""
        bad = []
        for i in range(len(supports)):
            for j in range(i, len(supports)):
                region = 0
                for q in supports[i] + supports[j]:
                    region |= 1 << q
                if not self.code.is_correctable_region(region):
                    bad.append((supports[i], supports[j]))
        return len(bad) == 0, bad

    # -- estimation --------------------------------------------------------
    def estimate_logical_moments(self, E_S: np.ndarray) -> np.ndarray:
        """Solve the log-linear system (Eq. (17)) in least squares and predict
        the logical moments E_L = exp(D_L log F).

        When the rank condition (Eq. (19)) holds, every row of D_L lies in the
        row span of D_S, so the prediction is independent of the (possibly
        non-unique) solution for log F."""
        logE = np.log(np.clip(E_S, 1e-12, None))
        x, *_ = np.linalg.lstsq(self.D_S, logE, rcond=None)
        return np.exp(self.D_L @ x)

    # -- logical channel reconstruction -------------------------------------
    def logical_channel(self, E_L: np.ndarray) -> dict:
        """P_L(e) = 4^{-n} sum_{l in L} <l, e> E(l)  (inverse Fourier of E * Phi_L).

        Returns coset probabilities Q[(syndrome, logical class)] = |S| * P_L(rep),
        i.e. the decoder-independent logical channel of Eq. (5)."""
        code = self.code
        pure, logical_reps = code.coset_representatives()
        Q: dict[tuple[int, Pauli], float] = {}
        norm = 4.0 ** code.n
        for sig, t in pure.items():
            for lbar in logical_reps:
                e = mul(t, lbar)
                val = sum(
                    bichar(l, e) * El for l, El in zip(code.logicals, E_L)
                )
                Q[(sig, lbar)] = len(code.stabilizers) * val / norm
        return Q


def true_logical_channel(code: StabilizerCode, dist: dict[Pauli, float]) -> dict:
    """Ground truth via Eq. (5): P_L(e) = |S|^{-1} sum_s P(es), summed over cosets."""
    pure, logical_reps = code.coset_representatives()
    Q: dict[tuple[int, Pauli], float] = {}
    for sig, t in pure.items():
        for lbar in logical_reps:
            e = mul(t, lbar)
            Q[(sig, lbar)] = sum(dist.get(mul(e, s), 0.0) for s in code.stabilizers)
    return Q


def tv_distance(Q1: dict, Q2: dict) -> float:
    return 0.5 * sum(abs(Q1[k] - Q2[k]) for k in Q1)
