"""Demo 3 -- The paper's punchline: correlated noise on the rotated d=3
surface code [[9,1,3]].

The surface code has pure distance 2 (weight-2 boundary stabilizers), so the
earlier result [Wagner et al., Quantum 6, 809 (2022) = Ref. 26] cannot
guarantee estimation of the *physical* channel for correlations across >= 2
qubits.  Theorem 1 of this paper only needs the noise to be *correctable*
(Definition 1): if the supports (here: all single qubits + two-qubit
correlations) have correctable unions, the *logical* channel is identifiable.

Expected outcome:
  physical channel: NOT identifiable  (rank(D_S) < |Gamma'|)
  logical channel:  identifiable      (Eq. 19 holds) and estimated correctly.
"""

import numpy as np

from pauli_learning import (
    CorrelatedNoise,
    LocalChannel,
    LogicalEstimator,
    StabilizerCode,
    true_logical_channel,
    tv_distance,
    wht,
)

rng = np.random.default_rng(3)

# rotated d=3 surface code, qubits in a 3x3 grid (row-major):
#   0 1 2
#   3 4 5
#   6 7 8
SURFACE = [
    "IXXIIIIII",  # X: top boundary (1,2)
    "XXIXXIIII",  # X: bulk (0,1,3,4)
    "IIIIXXIXX",  # X: bulk (4,5,7,8)
    "IIIIIIXXI",  # X: bottom boundary (6,7)
    "ZIIZIIIII",  # Z: left boundary (0,3)
    "IIIZZIZZI",  # Z: bulk (3,4,6,7)
    "IZZIZZIII",  # Z: bulk (1,2,4,5)
    "IIIIIZIIZ",  # Z: right boundary (5,8)
]
code = StabilizerCode(SURFACE)
print(f"Code: rotated surface code [[{code.n},{code.k},{code.distance()}]]")
print(f"  pure distance = {code.pure_distance()}  "
      "(=> Ref.[26] cannot even guarantee physical estimation for 2-qubit correlations)")

# --- noise: single-qubit noise on 7 qubits + two correlated pairs -------------
# Qubits 2 and 6 are noiseless: on this small d=3 code, every (pair, single)
# combination would otherwise complete a weight-3 logical string; larger-
# distance codes admit correlated noise on all qubits (cf. toric code example
# in the paper).
noisy_singles = [q for q in range(code.n) if q not in (2, 6)]
pairs = [(0, 1), (7, 8)]
supports = [(q,) for q in noisy_singles] + pairs
channels = (
    [LocalChannel.random((q,), error_rate=0.03, rng=rng) for q in noisy_singles]
    + [LocalChannel.random(p, error_rate=0.02, rng=rng) for p in pairs]
)
noise = CorrelatedNoise(channels)

est = LogicalEstimator(code, supports)
ok, bad = est.check_definition_1(supports)
print(f"\nDefinition 1 (all unions of supports correctable): {ok}")
if not ok:
    print("  non-correctable unions:", bad)
report = est.rank_report()
print("Rank report:", report)
print(f"  -> physical channel identifiable: {report['physical_identifiable']}"
      f"   (rank {report['rank_D_S']} < {report['n_params']} parameters)")
print(f"  -> logical channel identifiable (Eq. 19): {report['logical_identifiable']}")

# --- estimation ----------------------------------------------------------------
E_S = np.array([noise.moment(s) for s in code.stabilizers])
E_L_true = np.array([noise.moment(l) for l in code.logicals])
E_L_est = est.estimate_logical_moments(E_S)
print(f"\n[exact statistics] max |E_L_est - E_L_true| = "
      f"{np.max(np.abs(E_L_est - E_L_true)):.3e}")

Q_true = true_logical_channel(code, noise.full_distribution())
Q_est = est.logical_channel(E_L_est)
print(f"[exact statistics] TV(P_L_est, P_L_true)     = {tv_distance(Q_true, Q_est):.3e}")

print("\nConvergence with sampled syndromes:")
for shots in (10**4, 10**5, 10**6):
    q_emp = noise.sample_syndromes(code, shots, rng)
    E_L_emp = est.estimate_logical_moments(wht(q_emp))
    Q_emp = est.logical_channel(E_L_emp)
    print(f"  [{shots:>7} rounds]  TV(P_L_est, P_L_true) = "
          f"{tv_distance(Q_true, Q_emp):.3e}")

print("\n=> Correlated noise beyond the pure distance: the physical channel is")
print("   underdetermined by syndromes, yet the logical channel -- everything a")
print("   decoder needs (Eq. 4) -- is recovered exactly.  This is Theorem 1.")
