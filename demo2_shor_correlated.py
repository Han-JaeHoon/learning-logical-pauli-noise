"""Demo 2 -- Boundaries of Theorem 1, on the [[9,1,3]] Shor code.

Theorem 1: the logical channel is identifiable *as long as the code can
correct the noise* (Definition 1).  This demo probes both sides:

  (A0) baseline: independent single-qubit noise -> identifiable;
  (A)  within-block two-qubit correlations: the Shor code CANNOT correct
       these (X0X1 and X2 share a syndrome but differ by a logical), the
       hypothesis of the theorem fails -- and identifiability indeed fails;
  (B)  negative control: invisible logical noise on supp(Z_L);
  (C)  'up to logical equivalence': two different physical channels with the
       same syndrome statistics AND the same logical channel.
"""

import numpy as np

from pauli_learning import (
    SHOR,
    CorrelatedNoise,
    LocalChannel,
    LogicalEstimator,
    StabilizerCode,
    from_str,
    true_logical_channel,
    tv_distance,
    wht,
)
from pauli_learning.pauli import IDENTITY

rng = np.random.default_rng(11)

code = StabilizerCode(SHOR)
print(f"Code: [[{code.n},{code.k},{code.distance()}]] Shor code")
print(f"  pure distance = {code.pure_distance()}  "
      f"(=> Ref.[26] physical estimation requires correlations < {code.pure_distance()})")

# ==========================================================================
print("\n" + "=" * 74)
print("(A0) Baseline: independent single-qubit noise")
print("=" * 74)
singles = [(q,) for q in range(code.n)]
est0 = LogicalEstimator(code, singles)
print("Definition 1 satisfied:", est0.check_definition_1(singles)[0])
print("Rank report:", est0.rank_report())

# ==========================================================================
print("\n" + "=" * 74)
print("(A) Within-block pair correlations: noise the Shor code cannot correct")
print("=" * 74)

pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]  # within-block pairs
supports = [(q,) for q in range(code.n)] + pairs
channels = (
    [LocalChannel.random((q,), error_rate=0.03, rng=rng) for q in range(code.n)]
    + [LocalChannel.random(p, error_rate=0.02, rng=rng) for p in pairs]
)
noise = CorrelatedNoise(channels)

est = LogicalEstimator(code, supports)
ok, bad = est.check_definition_1(supports)
print(f"Definition 1 satisfied: {ok}"
      + ("" if ok else f"   (non-correctable unions, e.g. {bad[:2]} -- Def.1 is only sufficient)"))
report = est.rank_report()
print(f"Rank report: {report}")
print(f"  -> physical channel identifiable: {report['physical_identifiable']}"
      f"   ({report['rank_D_S']} equations of rank vs {report['n_params']} parameters)")
print(f"  -> logical channel identifiable (Eq. 19): {report['logical_identifiable']}")

# estimation from exact syndrome statistics
E_S = np.array([noise.moment(s) for s in code.stabilizers])
E_L_true = np.array([noise.moment(l) for l in code.logicals])
E_L_est = est.estimate_logical_moments(E_S)
print(f"\n[exact statistics] max |E_L_est - E_L_true| = "
      f"{np.max(np.abs(E_L_est - E_L_true)):.3e}")

dist = noise.full_distribution()
Q_true = true_logical_channel(code, dist)
Q_est = est.logical_channel(E_L_est)
print(f"[exact statistics] TV(P_L_est, P_L_true)     = {tv_distance(Q_true, Q_est):.3e}")
print("=> the estimation error does NOT vanish even with perfect statistics:")
print("   X0X1 and X2 have the same syndrome but X0X1*X2 = X0X1X2 is a logical")
print("   operator, so their probabilities cannot be told apart.  The theorem's")
print("   hypothesis (correctable noise) fails, and so does identifiability.")

# ==========================================================================
print("\n" + "=" * 74)
print("(B) Negative control: noise on supp(Z_L) = {0,3,6} (uncorrectable)")
print("=" * 74)

ZL = from_str("ZIIZIIZII")
assert code.is_logical(ZL) and not code.is_stabilizer(ZL)

base = [LocalChannel.random((q,), error_rate=0.03, rng=rng) for q in range(code.n)]
extra = LocalChannel((0, 3, 6), {IDENTITY: 0.95, ZL: 0.05})  # invisible logical error
noise_A = CorrelatedNoise(base)
noise_B = CorrelatedNoise(base + [extra])

qA = noise_A.exact_syndrome_distribution(code)
qB = noise_B.exact_syndrome_distribution(code)
QA = true_logical_channel(code, noise_A.full_distribution())
QB = true_logical_channel(code, noise_B.full_distribution())
print(f"max |q_A(sigma) - q_B(sigma)|  = {np.max(np.abs(qA - qB)):.3e}   (identical syndromes)")
print(f"TV(P_L^A, P_L^B)               = {tv_distance(QA, QB):.3e}   (different logical channels!)")

est_bad = LogicalEstimator(code, [(q,) for q in range(code.n)] + [(0, 3, 6)])
rep_bad = est_bad.rank_report()
print(f"Rank condition (Eq. 19) with support (0,3,6): "
      f"logical_identifiable = {rep_bad['logical_identifiable']}")
print("=> as the theorem requires: noise the code cannot correct cannot be learned.")

# ==========================================================================
print("\n" + "=" * 74)
print("(C) Estimation 'up to logical equivalence': Z_0 vs Z_1 noise")
print("=" * 74)

# Z_0 and Z_1 differ by the stabilizer Z_0 Z_1 -> logically equivalent errors.
p = 0.04
ch1 = LocalChannel((0, 1), {IDENTITY: 1 - p, from_str("ZIIIIIIII"): p})
ch2 = LocalChannel((0, 1), {IDENTITY: 1 - p, from_str("IZIIIIIII"): p})
rest = [LocalChannel.random((q,), error_rate=0.03, rng=rng) for q in range(2, code.n)]
noise_1 = CorrelatedNoise([ch1] + rest)
noise_2 = CorrelatedNoise([ch2] + rest)

q1 = noise_1.exact_syndrome_distribution(code)
q2 = noise_2.exact_syndrome_distribution(code)
d1 = noise_1.full_distribution()
d2 = noise_2.full_distribution()
phys_tv = 0.5 * sum(abs(d1.get(e, 0) - d2.get(e, 0)) for e in set(d1) | set(d2))
Q1 = true_logical_channel(code, d1)
Q2 = true_logical_channel(code, d2)
print(f"TV(physical P^1, P^2)          = {phys_tv:.3e}   (different physical channels)")
print(f"max |q_1(sigma) - q_2(sigma)|  = {np.max(np.abs(q1 - q2)):.3e}   (same syndromes)")
print(f"TV(P_L^1, P_L^2)               = {tv_distance(Q1, Q2):.3e}   (same logical channel)")
print("=> physically different but logically equivalent noise is indistinguishable --")
print("   and for optimal decoding (Eq. 4) that difference never mattered.")
