# KKTSimplex: Verifiable Epistemic Calibration & Autonomous Execution Governance

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **KKTSimplex** provides a mathematically proven epistemic calibration engine and deterministic execution governance layer for autonomous, multi-phase AI agents.

---

## 1. The Problem

Autonomous multi-phase coding agents systematically break down on complex engineering tasks due to two foundational pathologies:

1. **Epistemic Fragility & Sycophantic Capitulation:**
   Post-training alignment (RLHF / standard DPO) rewards conversational agreeableness. When challenged by a hostile user or auditor (*"Your implementation is completely broken, revert it!"*), uncalibrated models suffer epistemic collapse—they apologize and delete working implementations even when their code passed all tests.
2. **Context Drift & Lazy Shortcuts:**
   Over long execution trajectories, attention on the original requirement specifications dilutes. When encountering challenging edge cases, agents hallucinate completion by taking lazy shortcuts (`# TODO: implement edge case`, stubbed mocks) and self-certify `DONE` without deterministic verification.

---

## 2. Core Architecture

KKTSimplex unifies **convex optimization** (closed-form bounded simplex projection), **differential deterministic execution gates**, and **immutable spec invariant contracts** into a unified runtime:

```
                            ┌──────────────────────────────────────────────┐
                            │          Immutable Spec Invariant           │
                            │           (SpecAnchor Contract)              │
                            └──────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                            ┌──────────────────────────────────────────────┐
                            │        Execution-Guided Tree Search          │
                            │             (Calibrated MCTS)                │
                            └──────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                            ┌──────────────────────────────────────────────┐
                            │         Deterministic Execution Gate         │
                            │        (Baseline vs Post-Diff Shell)         │
                            │     PASS | FAIL_INTRODUCED_BY_DIFF | ENV     │
                            └──────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                            ┌──────────────────────────────────────────────┐
                            │        Convex Epistemic Trust Engine         │
                            │     (Water-Filling Simplex Projection)       │
                            │        w ∈ [floor, ceiling], ∑w = 1          │
                            └──────────────────────┬───────────────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
            [Verified PASS: w >= 0.70]                          [Regression or Broken Test]
                Status: DONE                                      Status: REVERT_AND_PARK
       (Rejects Hostile Capitulation)                       (Atomic Git Rollback Triggered)
```

---

## 3. Mathematical Foundations

### Bounded Simplex Projection with Box Constraints
Rather than relying on unconstrained softmax or heuristic clamping, KKTSimplex projects agent belief vectors onto the bounded probability simplex:
$$\mathcal{W} = \left\{ \mathbf{w} \in \mathbb{R}^N : \sum_{i=1}^N w_i = 1, \quad \text{floor} \le w_i \le \text{effective\_ceiling} \right\}$$
where $\text{effective\_ceiling} = \max(\text{ceiling}, 1/N)$.

The convex projection problem:
$$\min_{\mathbf{w} \in \mathcal{W}} \frac{1}{2} \|\mathbf{w} - \mathbf{x}\|_2^2$$
is solved in closed form using an exact **water-filling algorithm** on sorted breakpoints $\theta \in \{x_i - \text{upper}, x_i - \text{floor}\}$ in $\mathcal{O}(N \log N)$ time, guaranteeing zero numerical drift.

### Formal KKT Optimality Proof Certification
KKTSimplex includes a formal Karush-Kuhn-Tucker (KKT) proof generator (`verify_kkt_optimality`) that certifies every projection:
1. **Primal Feasibility:** $\min_i w_i \ge \text{floor} - 10^{-7}$, $\max_i w_i \le \text{upper} + 10^{-7}$, $|\sum w_i - 1| \le 10^{-7}$.
2. **Stationarity:** $\max_i |w_i - x_i + \theta - \lambda_i^{\text{low}} + \lambda_i^{\text{high}}| \le 10^{-7}$.
3. **Complementary Slackness:** $\max_i [\lambda_i^{\text{low}}(w_i - \text{floor})] \le 10^{-7}$ and $\max_i [\lambda_i^{\text{high}}(\text{upper} - w_i)] \le 10^{-7}$.
4. **Dual Feasibility:** $\lambda_i^{\text{low}} \ge 0, \lambda_i^{\text{high}} \ge 0$.

### Bayesian Log-Domain Evidence Accumulation
Evidence update in logit space:
$$\text{logit}_i = \log S_i + \alpha \frac{v_i}{n_i} - \beta \frac{h_i}{n_i}$$
$$\mathbf{w}_{t+1} = \text{Project}_{\mathcal{W}}(\text{softmax}(\mathbf{logit}))$$

* **Neutral Evidence Identity:** When $v = h = 0$, $\mathbf{w}_{t+1} \equiv \mathbf{w}_t$ (exact identity, drift $< 10^{-14}$).
* **Verbosity Invariance:** Scaling support $v_i$, harm $h_i$, and claim counts $n_i$ by factor $k$ produces identical trust weights, neutralizing claim duplication spam.

---

> Implements Bayesian log-domain trust updates bounded to [floor, ceiling].
