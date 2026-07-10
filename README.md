# Learning Logical Pauli Noise in Quantum Error Correction — Numerical Reproduction

*[한국어 README](README_kor.md)*

A numerical reproduction of the central results of

> T. Wagner, H. Kampermann, D. Bruß, M. Kliesch,
> *Learning Logical Pauli Noise in Quantum Error Correction*,
> Phys. Rev. Lett. **130**, 200601 (2023).

> **Theorem 1.** If a Pauli channel P is correctable (in the sense of
> Definition 1), then its **logical channel** can be estimated up to logical
> equivalence from the syndrome measurements of a stabilizer code.

## Layout

```
pauli_learning/
  pauli.py      Effective Pauli group (binary symplectic), bicharacter <a,e> (Eq. 1), substring order b <= a
  code.py       Stabilizer code: S, L = S^perp (Eq. 2), syndromes, correctable regions, (pure) distance
  noise.py      Correlated Pauli noise P = *_gamma P_gamma (Eq. 12), exact moments (Eq. 6, 9), syndrome sampling
  estimate.py   Gamma' (Eq. 15), D_S/D_L (Eq. 18), rank condition (Eq. 19),
                log-linear estimator (Eq. 17), logical-channel reconstruction (Eq. 5, 11)
demo1_five_qubit.py         [[5,1,3]]: full pipeline + 1/sqrt(N) convergence
demo2_shor_correlated.py    [[9,1,3]] Shor: boundary of the theorem (uncorrectable noise -> not learnable)
demo3_surface_correlated.py [[9,1,3]] surface: the punchline (physical NOT identifiable, logical IS)
```

The pipeline follows the paper directly:

1. Collect syndromes over QEC rounds -> empirical syndrome distribution q(σ)
2. Walsh–Hadamard transform -> stabilizer moments E(s) = F[P](s) (Eq. 6)
3. Solve log E(s) = Σ_{a≤s, a∈Γ'} log F(a) in least squares (Eq. 17)
4. Predict logical moments E(l) = exp(D_L log F) -> reconstruct logical channel P_L (inverse of Eq. 11)

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
.venv/bin/python demo1_five_qubit.py
.venv/bin/python demo2_shor_correlated.py
.venv/bin/python demo3_surface_correlated.py
```

## Summary of results

| Demo | Code | Noise | Def. 1 | Rank cond. (Eq. 19) | Outcome |
|---|---|---|---|---|---|
| 1 | [[5,1,3]] | independent single-qubit | ✓ | logical ✓ / physical ✓ | error ~1e-16 from exact statistics; TV ∝ 1/√N when sampled |
| 2(A0) | Shor [[9,1,3]] | independent single-qubit | ✓ | logical ✓ / physical ✗ | Z0↔Z1 gauge freedom: physical undetermined to begin with |
| 2(A) | Shor | in-block 2-qubit correlation | ✗ | logical ✗ | X0X1↔X2 share a syndrome but differ by a logical → error plateau |
| 2(B) | Shor | noise on supp(Z̄)={0,3,6} | ✗ | logical ✗ | explicit pair of models: same syndromes, different logical channels |
| 2(C) | Shor | Z0 vs Z1 | — | — | physically different but identical syndromes AND logical channel |
| 3 | surface [[9,1,3]] | single-qubit + correlated pairs | ✓ | **logical ✓ / physical ✗** | correlations beyond pure distance (=2): logical channel recovered exactly |

- **Demo 1** — direct verification of the theorem: from exact syndrome
  statistics the logical channel is recovered to machine precision, and with
  finite samples the total-variation error converges as 1/√N
  (`fig_demo1_convergence.png`).
- **Demo 2** — when the theorem's hypothesis ("the code can correct the
  noise") fails, estimation genuinely becomes impossible: if two error events
  differing by a logical operator share the same syndrome, the logical channel
  is undetermined even with perfect statistics.
- **Demo 3** — the paper's punchline: for correlated noise beyond the pure
  distance, the **physical channel is underdetermined by syndromes
  (rank 37 < 39), yet the logical channel — all a decoder needs — is
  recovered exactly (error ~1e-15).**

## Correspondence with the paper

| Paper | Code |
|---|---|
| Eq. (1) bicharacter | `pauli.bichar` |
| Eq. (2) L = S^perp | `StabilizerCode.logicals` |
| Eq. (5) logical channel | `estimate.true_logical_channel` |
| Eq. (6) Fourier / moments | `LocalChannel.moment`, `noise.wht` |
| Eq. (9), (12) convolution↔product | `CorrelatedNoise.moment` |
| Eq. (15) Gamma' | `estimate.gamma_prime` |
| Eq. (17) polynomial system | `LogicalEstimator.estimate_logical_moments` |
| Eq. (18) D matrices | `estimate.coefficient_matrix` |
| Eq. (19) rank condition | `LogicalEstimator.rank_report` |
| Definition 1 | `LogicalEstimator.check_definition_1` |
| Theorem 1 | demos 1 & 3 (positive), demo 2 (boundary) |

### A note on distance and correlated noise

On the small d=3 codes, "single-qubit noise on every qubit + any 2-qubit
correlation" always violates Definition 1 (verified by exhaustive search):
weight-3 logical operators are dense enough that any (pair, single)
combination completes a logical string. Demo 3 therefore leaves two qubits
noiseless. This mirrors the paper's toric-code remark that correctable
regions (e.g. a square of side ≤ d−1) are what admit correlated noise —
**learning correlated noise needs distance headroom**, which shows up
quantitatively already at small sizes.

Note: following the paper's phenomenological setting, measurements are assumed
perfect (measurement errors extend to data-syndrome codes — see the paper's
Supplemental Material).
