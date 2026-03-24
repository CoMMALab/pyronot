# Matrix-Free TrajOpt: LS and SCO

This note explains how to avoid dense Jacobian materialization in trajectory optimization.

The short version:
- LS TrajOpt now uses a matrix-free Levenberg-Marquardt (LM) update solved with conjugate gradients (CG).
- SCO TrajOpt already avoids dense Jacobian materialization in its inner solve by using first-order L-BFGS steps; it does not currently run LM+CG in the inner loop.

## Why Dense Jacobians Are Expensive

For a trajectory with:
- batch size B
- timesteps T
- dof n_act

we optimize a flattened state:

x in R^(n), n = T * n_act

The stacked residual r(x) has dimension m (smoothness + trust + limits + collision + endpoint terms), and typically m >> n.

A dense LM step builds:
- J = dr/dx in R^(m x n)
- A = J^T J + lambda I in R^(n x n)

Then solves:

(J^T J + lambda I) * delta = -J^T r

Materializing J and forming J^T J each iteration is often the main bottleneck.

## Matrix-Free Normal Equations

Instead of building J explicitly, we only need products with J and J^T.

Define the normal-equation operator:

A(v) = J^T (J v) + lambda v

Each CG iteration only needs A(v), which can be computed using:
- JVP (forward linearization) for J v
- transpose-JVP / VJP for J^T u

No dense Jacobian matrix is built.

## LS Path (Current)

The LS inner loop is now:
1. Build a linearized residual map around the current x.
2. Define matrix-free operator A(v) = J^T(Jv) + lambda v.
3. Solve A(delta) = -J^T r with truncated CG.
4. Clamp delta, run short alpha line-search, accept/reject, update lambda.
5. Early-stop inner iterations after repeated non-improving steps.

Practical effects:
- removes dense J materialization from the LS inner loop
- removes dense n x n direct linear solves
- reduces runtime significantly while preserving robot interchangeability

Robot interchangeability is preserved because all derivatives come from automatic differentiation over the residual function, not from robot-specific analytic Jacobian code.

## SCO Path (Current)

SCO currently follows a different inner solve strategy:
- linearize collision constraints per outer iteration
- solve convexified subproblem with L-BFGS + line search

This is also matrix-free in the inner loop (no dense Hessian/Jacobian factorization), but it is not LM+CG.

## If You Want SCO and LS on One LM+CG Core

A unified design is possible:
- keep SCO outer collision linearization
- replace SCO inner L-BFGS with the same matrix-free LM+CG core used in LS
- retain shared residual blocks (smoothness, limits, trust, collision, endpoint)

Benefits:
- one numerics core for both solvers
- less duplicated optimization code
- identical matrix-free behavior across methods

Trade-off:
- may reduce some of SCO's current speed advantage from its specialized first-order inner updates

## Minimal Pseudocode (Matrix-Free LM+CG)

```text
for outer in range(n_outer):
    linearize collision at x_k   # d_k, J_k or equivalent linearized collision residual

    x = x_k
    lambda = lambda_init
    for inner in range(n_lm):
        r, jvp = linearize(residual_fn, x)

        # matrix-free transpose
        JT(u) = transpose(jvp)(u)

        # normal operator for CG
        A(v) = JT(jvp(v)) + lambda * v
        b = -JT(r)

        delta = cg(A, b, n_cg, tol)
        delta = clip(delta)

        x_trial = line_search(x, delta)
        if improved(x_trial):
            x = x_trial
            lambda *= 0.5
        else:
            lambda *= 3.0

    x_k = x
```

## Summary

- Dense LM is accurate but expensive due to Jacobian materialization and dense solves.
- Matrix-free LM+CG keeps the same objective structure while cutting memory traffic and linear algebra cost.
- LS in this codebase uses that matrix-free LM+CG approach now.
- SCO currently stays first-order in the inner loop, but can be migrated to the same matrix-free LM+CG core if desired.
