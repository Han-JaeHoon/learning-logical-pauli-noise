"""Demo 1 -- Full pipeline on the [[5,1,3]] code with independent single-qubit noise.

Reproduces the paper's claim end to end:
  syndrome statistics -> stabilizer moments E_S -> log-linear system (Eq. 17)
  -> logical moments E_L -> logical channel P_L, compared against brute force.
"""

import numpy as np

from pauli_learning import (
    FIVE_QUBIT,
    CorrelatedNoise,
    LocalChannel,
    LogicalEstimator,
    StabilizerCode,
    to_str,
    true_logical_channel,
    tv_distance,
    wht,
)

rng = np.random.default_rng(7)

code = StabilizerCode(FIVE_QUBIT)
print(f"Code: {code}")
print(f"  distance d = {code.distance()}, pure distance = {code.pure_distance()}")
print(f"  |S| = {len(code.stabilizers)}, |L| = {len(code.logicals)}")

# --- noise: independent asymmetric single-qubit channels ---------------------
supports = [(q,) for q in range(code.n)]
channels = [LocalChannel.random((q,), error_rate=0.06 + 0.02 * q, rng=rng)
            for q in range(code.n)]
noise = CorrelatedNoise(channels)

est = LogicalEstimator(code, supports)
ok, bad = est.check_definition_1(supports)
print(f"\nDefinition 1 (unions of supports correctable): {ok}")
print("Rank report (Eq. 19):", est.rank_report())

# --- ground truth -------------------------------------------------------------
dist = noise.full_distribution()
E_S_exact = np.array([noise.moment(s) for s in code.stabilizers])
E_L_exact = np.array([noise.moment(l) for l in code.logicals])

# sanity: product-form moments == brute-force Fourier transform
q_exact = noise.exact_syndrome_distribution(code)
assert np.allclose(wht(q_exact), E_S_exact), "moment consistency check failed"

# --- (a) estimation from EXACT syndrome statistics ----------------------------
E_L_from_exact = est.estimate_logical_moments(E_S_exact)
err_exact = np.max(np.abs(E_L_from_exact - E_L_exact))
print(f"\n[exact statistics]  max |E_L_est - E_L_true| = {err_exact:.3e}")

Q_true = true_logical_channel(code, dist)
Q_exact = est.logical_channel(E_L_from_exact)
print(f"[exact statistics]  TV(P_L_est, P_L_true)      = {tv_distance(Q_true, Q_exact):.3e}")

# --- (b) estimation from SAMPLED syndromes (finite shots) ---------------------
print("\nConvergence with number of QEC rounds (sampled syndromes):")
print(f"{'shots':>10} | {'max moment err':>15} | {'TV(P_L est, true)':>18}")
shot_list = [10**3, 10**4, 10**5, 10**6]
tv_list = []
for shots in shot_list:
    q_emp = noise.sample_syndromes(code, shots, rng)
    E_S_emp = wht(q_emp)
    E_L_emp = est.estimate_logical_moments(E_S_emp)
    Q_emp = est.logical_channel(E_L_emp)
    tv = tv_distance(Q_true, Q_emp)
    tv_list.append(tv)
    print(f"{shots:>10} | {np.max(np.abs(E_L_emp - E_L_exact)):>15.3e} | {tv:>18.3e}")

# --- (c) show the logical channel table for a few syndromes -------------------
print("\nConditional logical class probabilities P(logical class | syndrome)")
print("(estimated with 10^6 rounds vs. ground truth; first 4 syndromes)")
pure, logical_reps = code.coset_representatives()
labels = [to_str(l, code.n) for l in logical_reps]
print(f"{'syndrome':>9} | " + " | ".join(f"{lb:^17}" for lb in labels))
print(f"{'':>9} | " + " | ".join(f"{'est':>8} {'true':>8}" for _ in labels))
for sig in sorted(pure)[:4]:
    tot_e = sum(Q_emp[(sig, l)] for l in logical_reps)
    tot_t = sum(Q_true[(sig, l)] for l in logical_reps)
    row = " | ".join(
        f"{Q_emp[(sig, l)] / tot_e:>8.4f} {Q_true[(sig, l)] / tot_t:>8.4f}"
        for l in logical_reps
    )
    print(f"{sig:>9} | {row}")

# --- convergence plot ----------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(shot_list, tv_list, "o-", label="TV(estimated $P_L$, true $P_L$)")
guide = tv_list[0] * np.sqrt(shot_list[0]) / np.sqrt(np.array(shot_list, float))
ax.loglog(shot_list, guide, "k--", alpha=0.5, label=r"$\propto 1/\sqrt{N}$")
ax.set_xlabel("QEC rounds (syndrome samples)")
ax.set_ylabel("total variation distance")
ax.set_title("[[5,1,3]] code: logical channel estimation from syndromes")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("fig_demo1_convergence.png", dpi=150)
print("\nSaved: fig_demo1_convergence.png")
