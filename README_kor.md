# Learning Logical Pauli Noise in Quantum Error Correction — 수치 재연

*[English README](README.md)*

Wagner, Kampermann, Bruß, Kliesch, *Phys. Rev. Lett.* **130**, 200601 (2023)
([PDF](../Learning%20Logical%20Pauli%20Noise%20in%20Quantum%20Error%20Correction.pdf))의
핵심 결과를 수치적으로 재연한다.

> **Theorem 1.** Pauli 채널 P가 (Definition 1의 의미로) correctable 하면,
> 그 **논리 채널**은 stabilizer 코드의 신드롬 측정만으로 논리적 동치까지
> 유일하게 추정할 수 있다.

## 구조

```
pauli_learning/
  pauli.py      유효 Pauli 군 (binary symplectic), bicharacter <a,e> (Eq. 1), substring 순서
  code.py       stabilizer 코드: S, L = S^perp (Eq. 2), 신드롬, correctable region, (pure) distance
  noise.py      상관 Pauli 노이즈 P = *_gamma P_gamma (Eq. 12), 정확한 모멘트 (Eq. 6, 9), 신드롬 샘플링
  estimate.py   Gamma' (Eq. 15), D_S/D_L (Eq. 18), rank 조건 (Eq. 19),
                log-선형 추정 (Eq. 17), 논리 채널 재구성 (Eq. 5, 11)
demo1_five_qubit.py         [[5,1,3]]: 전체 파이프라인 + 1/sqrt(N) 수렴
demo2_shor_correlated.py    [[9,1,3]] Shor: 정리의 경계 (정정 불가능 노이즈 → 추정 불가능)
demo3_surface_correlated.py [[9,1,3]] surface: 펀치라인 (물리 X, 논리 O)
```

핵심 파이프라인 (논문 그대로):

1. QEC 라운드마다 신드롬 수집 → 경험적 신드롬 분포 q(σ)
2. Walsh–Hadamard 변환 → stabilizer 모멘트 E(s) = F[P](s) (Eq. 6)
3. log E(s) = Σ_{a≤s, a∈Γ'} log F(a) 최소제곱 풀이 (Eq. 17)
4. 논리 모멘트 E(l) = exp(D_L log F) → 논리 채널 P_L 재구성 (Eq. 11 역변환)

## 실행

```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
.venv/bin/python demo1_five_qubit.py
.venv/bin/python demo2_shor_correlated.py
.venv/bin/python demo3_surface_correlated.py
```

## 결과 요약

| 데모 | 코드 | 노이즈 | Def. 1 | rank 조건 (Eq. 19) | 결과 |
|---|---|---|---|---|---|
| 1 | [[5,1,3]] | 독립 단일큐빗 | ✓ | 논리 ✓ / 물리 ✓ | 정확 통계에서 오차 ~1e-16, 샘플링 시 TV ∝ 1/√N |
| 2(A0) | Shor [[9,1,3]] | 독립 단일큐빗 | ✓ | 논리 ✓ / 물리 ✗ | Z0↔Z1 게이지 자유도: 물리는 원래 결정 불가 |
| 2(A) | Shor | 블록 내 2큐빗 상관 | ✗ | 논리 ✗ | X0X1↔X2가 같은 신드롬·다른 논리 클래스 → 오차 플래토 |
| 2(B) | Shor | supp(Z̄)={0,3,6} 위 노이즈 | ✗ | 논리 ✗ | 신드롬 동일·논리 채널 상이한 두 모델 명시적 구성 |
| 2(C) | Shor | Z0 vs Z1 | — | — | 물리적으로 다르지만 신드롬·논리 채널 완전 동일 |
| 3 | surface [[9,1,3]] | 단일큐빗 + 상관 pair | ✓ | **논리 ✓ / 물리 ✗** | pure distance(=2) 이상의 상관에도 논리 채널 정확 복원 |

- **데모 1** — 정리의 직접 검증: 정확한 신드롬 통계로부터 논리 채널이 기계 정밀도로
  복원되고, 유한 샘플에서는 1/√N으로 수렴한다 (`fig_demo1_convergence.png`).
- **데모 2** — 정리의 가정("코드가 노이즈를 정정할 수 있음")이 깨지면 실제로 추정이
  불가능함을 보인다: 논리 연산자만큼 차이 나는 오류쌍이 같은 신드롬을 가지면
  완벽한 통계로도 논리 채널이 결정되지 않는다.
- **데모 3** — 논문의 펀치라인: pure distance를 넘는 상관 노이즈에서 **물리 채널은
  신드롬으로 미결정(rank 37 < 39)이지만, 디코더에 필요한 전부인 논리 채널은
  정확히 (오차 ~1e-15) 복원**된다.

## 논문과의 대응

| 논문 | 코드 |
|---|---|
| Eq. (1) bicharacter | `pauli.bichar` |
| Eq. (2) L = S^perp | `StabilizerCode.logicals` |
| Eq. (5) 논리 채널 | `estimate.true_logical_channel` |
| Eq. (6) Fourier/모멘트 | `LocalChannel.moment`, `noise.wht` |
| Eq. (9), (12) 합성곱↔곱 | `CorrelatedNoise.moment` |
| Eq. (15) Gamma' | `estimate.gamma_prime` |
| Eq. (17) 다항 방정식계 | `LogicalEstimator.estimate_logical_moments` |
| Eq. (18) D 행렬 | `estimate.coefficient_matrix` |
| Eq. (19) rank 조건 | `LogicalEstimator.rank_report` |
| Definition 1 | `LogicalEstimator.check_definition_1` |
| Theorem 1 | 데모 1·3 (양성), 데모 2 (경계) |

주의: 논문의 phenomenological 설정 그대로 측정은 완벽하다고 가정한다
(측정 오류는 data-syndrome 코드로 확장 가능 — 논문 Supplemental 참조).
