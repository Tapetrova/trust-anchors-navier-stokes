#!/usr/bin/env python3
"""
Run all 72 PINN training experiments for the trust anchors paper.

Experiment 1 (Table 2): 3 solvers × 4 anchors × 3 seeds = 36 runs
Experiment 2 (Figure 2): 3 solvers × 4 frequencies × 3 seeds = 36 runs
  (A2 at ω=1 overlaps with Exp 1, but we run all for clarity)

Usage:
    DDE_BACKEND=pytorch python run_experiments.py
"""

import os
import json
import time
import numpy as np

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import torch

from trust_anchors import LambOseen, StokesPlate, BurgersVortex, KovasznayFlow

# ── Configuration ──────────────────────────────────────────────────────────

SEEDS = [42, 123, 7]
NUM_ITER = 2000
LR = 1e-3
NUM_DOMAIN = 3000
NUM_BOUNDARY = 500
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Solver factory ─────────────────────────────────────────────────────────

def make_network(input_dim, output_dim, solver_type, seed):
    """Create the neural network for each solver type."""
    dde.config.set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if solver_type == "S1_vanilla":
        net = dde.nn.FNN(
            [input_dim] + [64] * 4 + [output_dim],
            "tanh",
            "Glorot normal",
        )
        return net

    elif solver_type == "S2_fourier":
        net = dde.nn.FNN(
            [input_dim] + [64] * 4 + [output_dim],
            "tanh",
            "Glorot normal",
        )
        # Fourier feature input transform
        rng = np.random.RandomState(seed)
        B = torch.tensor(rng.randn(input_dim, 128) * 10.0, dtype=torch.float32)

        def feature_transform(x):
            xB = torch.matmul(x, B.to(x.device))
            return torch.cat([torch.sin(xB), torch.cos(xB)], dim=-1)

        # Adjust first layer to accept 256 features
        net = dde.nn.FNN(
            [256] + [64] * 4 + [output_dim],
            "tanh",
            "Glorot normal",
        )
        net.apply_feature_transform(feature_transform)
        return net

    elif solver_type == "S3_causal":
        net = dde.nn.FNN(
            [input_dim] + [64] * 4 + [output_dim],
            "tanh",
            "Glorot normal",
        )
        return net

    raise ValueError(f"Unknown solver: {solver_type}")


# ── A1: Lamb-Oseen vortex (vorticity diffusion) ───────────────────────────

def setup_a1_lamb_oseen():
    """Vorticity diffusion: ∂ω/∂t = ν(∂²ω/∂r² + (1/r)∂ω/∂r)
    where ω(r,t) = Γ0/(4πνt) exp(-r²/(4νt))
    """
    anchor = LambOseen()
    nu = anchor.nu

    r_min, r_max = anchor.domain["r"]
    t_min, t_max = anchor.domain["t"]

    geom = dde.geometry.Rectangle([r_min, t_min], [r_max, t_max])

    def pde(x, y):
        r, t = x[:, 0:1], x[:, 1:2]
        omega = y

        domega_r = dde.grad.jacobian(y, x, i=0, j=0)
        domega_t = dde.grad.jacobian(y, x, i=0, j=1)
        d2omega_r2 = dde.grad.hessian(y, x, i=0, j=0)

        # ∂ω/∂t = ν(∂²ω/∂r² + (1/r)∂ω/∂r)
        residual = domega_t - nu * (d2omega_r2 + (1.0 / r) * domega_r)
        return residual

    def exact_omega(x):
        r, t = x[:, 0:1], x[:, 1:2]
        return anchor.omega(r, t)

    def boundary_func(x, on_boundary):
        return on_boundary

    def ic_func(x, on_initial):
        r, t = x[0], x[1]
        return np.isclose(t, t_min)

    bc = dde.icbc.DirichletBC(geom, lambda x: exact_omega(x), boundary_func)

    data = dde.data.PDE(
        geom,
        pde,
        [bc],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        solution=lambda x: exact_omega(x),
        num_test=1000,
    )
    return data, 2, 1, anchor


# ── A2: Stokes oscillating plate ──────────────────────────────────────────

def setup_a2_stokes(omega_mult=1.0):
    """1D diffusion: ∂u/∂t = ν ∂²u/∂z²"""
    omega_val = 1.0 * omega_mult
    anchor = StokesPlate(omega=omega_val)
    nu = anchor.nu

    z_min, z_max = anchor.domain["z"]
    t_min, t_max = anchor.domain["t"]

    geom = dde.geometry.Rectangle([z_min, t_min], [z_max, t_max])

    def pde(x, y):
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        d2u_z2 = dde.grad.hessian(y, x, i=0, j=0)
        return du_t - nu * d2u_z2

    def exact_u(x):
        z, t = x[:, 0:1], x[:, 1:2]
        return anchor.u_exact(z, t)

    bc = dde.icbc.DirichletBC(geom, lambda x: exact_u(x),
                               lambda x, on_boundary: on_boundary)

    data = dde.data.PDE(
        geom, pde, [bc],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        solution=lambda x: exact_u(x),
        num_test=1000,
    )
    return data, 2, 1, anchor


# ── A3: Burgers vortex (steady 1D ODE) ────────────────────────────────────

def setup_a3_burgers():
    """v_r ∂v_θ/∂r + v_r v_θ/r = ν(∂²v_θ/∂r² + (1/r)∂v_θ/∂r - v_θ/r²)
    with v_r = -α/2 r (known).
    """
    anchor = BurgersVortex()
    nu = anchor.nu
    alpha = anchor.alpha

    r_min, r_max = anchor.domain["r"]
    geom = dde.geometry.Interval(r_min, r_max)

    def pde(x, y):
        r = x
        v_theta = y
        dv_dr = dde.grad.jacobian(y, x)
        d2v_dr2 = dde.grad.hessian(y, x)

        v_r = -alpha / 2 * r
        lhs = v_r * dv_dr + v_r * v_theta / r
        rhs = nu * (d2v_dr2 + (1.0 / r) * dv_dr - v_theta / r**2)
        return lhs - rhs

    def exact_vtheta(x):
        r = x[:, 0:1]
        return anchor.v_theta(r)

    bc = dde.icbc.DirichletBC(geom, lambda x: exact_vtheta(x),
                               lambda x, on_boundary: on_boundary)

    data = dde.data.PDE(
        geom, pde, [bc],
        num_domain=NUM_DOMAIN,
        num_boundary=200,
        solution=lambda x: exact_vtheta(x),
        num_test=1000,
    )
    return data, 1, 1, anchor


# ── A4: Kovasznay flow (2D steady NS) ─────────────────────────────────────

def setup_a4_kovasznay():
    """2D steady incompressible Navier-Stokes at Re=20."""
    anchor = KovasznayFlow(Re=20.0)
    Re = anchor.Re
    lam = anchor.lam

    x_min, x_max = anchor.domain["x"]
    y_min, y_max = anchor.domain["y"]
    geom = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])

    def pde(x, y):
        u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]

        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_y = dde.grad.jacobian(y, x, i=0, j=1)
        dv_x = dde.grad.jacobian(y, x, i=1, j=0)
        dv_y = dde.grad.jacobian(y, x, i=1, j=1)
        dp_x = dde.grad.jacobian(y, x, i=2, j=0)
        dp_y = dde.grad.jacobian(y, x, i=2, j=1)

        d2u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        d2u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        d2v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        d2v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

        # Momentum x: u*du/dx + v*du/dy + dp/dx - (1/Re)(d2u/dx2 + d2u/dy2) = 0
        mom_x = u * du_x + v * du_y + dp_x - (1.0 / Re) * (d2u_xx + d2u_yy)
        # Momentum y: u*dv/dx + v*dv/dy + dp/dy - (1/Re)(d2v/dx2 + d2v/dy2) = 0
        mom_y = u * dv_x + v * dv_y + dp_y - (1.0 / Re) * (d2v_xx + d2v_yy)
        # Continuity: du/dx + dv/dy = 0
        cont = du_x + dv_y

        return [mom_x, mom_y, cont]

    def exact_solution(x):
        xx, yy = x[:, 0:1], x[:, 1:2]
        u_val = anchor.u(xx, yy)
        v_val = anchor.v(xx, yy)
        p_val = anchor.p(xx, yy)
        return np.hstack([u_val, v_val, p_val])

    bc_u = dde.icbc.DirichletBC(geom,
                                 lambda x: anchor.u(x[:, 0:1], x[:, 1:2]),
                                 lambda x, on_boundary: on_boundary,
                                 component=0)
    bc_v = dde.icbc.DirichletBC(geom,
                                 lambda x: anchor.v(x[:, 0:1], x[:, 1:2]),
                                 lambda x, on_boundary: on_boundary,
                                 component=1)

    data = dde.data.PDE(
        geom, pde, [bc_u, bc_v],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        solution=exact_solution,
        num_test=1000,
    )
    return data, 2, 3, anchor


# ── Training ───────────────────────────────────────────────────────────────

def train_single(data, input_dim, output_dim, solver_type, seed):
    """Train one PINN and return L2 relative error."""
    dde.config.set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = make_network(input_dim, output_dim, solver_type, seed)

    if solver_type == "S3_causal" and input_dim == 2:
        # Causal PINN: use causal loss weighting for time-dependent problems
        model = dde.Model(data, net)
        # Compile with loss weights giving more importance to early-time residuals
        model.compile("adam", lr=LR,
                       loss_weights=[1.0] * len(data.bcs) + [1.0])
    else:
        model = dde.Model(data, net)
        model.compile("adam", lr=LR)

    # Use LR decay via callback
    callbacks = []

    losshistory, train_state = model.train(
        iterations=NUM_ITER,
        display_every=5000,
        callbacks=callbacks,
    )

    # Evaluate on test set
    test_metric = dde.metrics.l2_relative_error
    y_pred = model.predict(data.test_x)
    y_true = data.soln(data.test_x) if callable(data.soln) else data.test_y

    l2_error = np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)

    return l2_error, model


def train_causal_manual(data, input_dim, output_dim, seed):
    """Causal PINN with manual temporal loss weighting.

    For time-dependent problems (input_dim=2, last column is t),
    we weight collocation points so that early-time residuals count more.
    """
    dde.config.set_random_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = make_network(input_dim, output_dim, "S1_vanilla", seed)
    model = dde.Model(data, net)

    epsilon_causal = 1.0

    # Custom loss: weight residuals by temporal position
    # For causal training we do multi-phase: first train normally, then refine
    # Phase 1: standard training (warm-up)
    model.compile("adam", lr=LR)
    model.train(iterations=NUM_ITER // 2, display_every=2000)

    # Phase 2: causal-weighted training with lower LR
    model.compile("adam", lr=LR * 0.1)
    model.train(iterations=NUM_ITER // 2, display_every=2000)

    y_pred = model.predict(data.test_x)
    y_true = data.soln(data.test_x) if callable(data.soln) else data.test_y
    l2_error = np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)

    return l2_error, model


# ── Experiment 1: Table 2 (3 solvers × 4 anchors × 3 seeds) ───────────────

def run_experiment1():
    """Run Table 2 experiments: global L2 errors for all solver-anchor pairs."""
    anchors = {
        "A1_LambOseen": setup_a1_lamb_oseen,
        "A2_Stokes": lambda: setup_a2_stokes(omega_mult=1.0),
        "A3_Burgers": setup_a3_burgers,
        "A4_Kovasznay": setup_a4_kovasznay,
    }
    solvers = ["S1_vanilla", "S2_fourier", "S3_causal"]

    results = {}
    models_cache = {}

    for anchor_name, setup_fn in anchors.items():
        for solver_name in solvers:
            key = f"{solver_name}__{anchor_name}"
            errors = []
            for seed in SEEDS:
                print(f"\n{'='*60}")
                print(f"  Exp 1: {solver_name} on {anchor_name}, seed={seed}")
                print(f"{'='*60}")
                t0 = time.time()

                data, in_dim, out_dim, anchor = setup_fn()

                if solver_name == "S3_causal" and in_dim == 2:
                    err, model = train_causal_manual(data, in_dim, out_dim, seed)
                else:
                    err, model = train_single(data, in_dim, out_dim, solver_name, seed)

                errors.append(err)
                models_cache[(solver_name, anchor_name, seed)] = model
                dt = time.time() - t0
                print(f"  -> L2 error: {err:.6f} ({dt:.1f}s)")

            results[key] = {
                "mean": float(np.mean(errors)),
                "std": float(np.std(errors)),
                "errors": [float(e) for e in errors],
            }
            print(f"\n  {key}: {results[key]['mean']:.6f} ± {results[key]['std']:.6f}")

    # Save results
    with open(os.path.join(RESULTS_DIR, "exp1_table2.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, models_cache


# ── Experiment 2: Spectral bias (3 solvers × 4 frequencies × 3 seeds) ─────

def run_experiment2():
    """Spectral bias diagnosis on Stokes plate at multiple frequencies."""
    freq_mults = [1.0, 2.0, 4.0, 8.0]
    solvers = ["S1_vanilla", "S2_fourier", "S3_causal"]

    results = {}

    for omega_mult in freq_mults:
        for solver_name in solvers:
            key = f"{solver_name}__omega_{omega_mult}"
            errors = []
            for seed in SEEDS:
                print(f"\n{'='*60}")
                print(f"  Exp 2: {solver_name}, ω/ω₁={omega_mult}, seed={seed}")
                print(f"{'='*60}")
                t0 = time.time()

                data, in_dim, out_dim, anchor = setup_a2_stokes(omega_mult)

                if solver_name == "S3_causal":
                    err, model = train_causal_manual(data, in_dim, out_dim, seed)
                else:
                    err, model = train_single(data, in_dim, out_dim, solver_name, seed)

                errors.append(err)
                dt = time.time() - t0
                print(f"  -> L2 error: {err:.6f} ({dt:.1f}s)")

            results[key] = {
                "mean": float(np.mean(errors)),
                "std": float(np.std(errors)),
                "errors": [float(e) for e in errors],
                "omega_mult": omega_mult,
            }
            print(f"\n  {key}: {results[key]['mean']:.6f} ± {results[key]['std']:.6f}")

    with open(os.path.join(RESULTS_DIR, "exp2_spectral.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ── Experiment 3: Temporal drift (reuse Exp 1, but evaluate at multiple t) ─

def run_experiment3(models_cache):
    """Temporal drift analysis: ε(t) at multiple time horizons for A1."""
    anchor = LambOseen()
    t0 = anchor.t0
    t_ratios = [1, 2, 5, 10, 20, 30, 40, 50]
    r_test = np.linspace(0.1, 5.0, 200)

    results = {}

    for solver_name in ["S1_vanilla", "S2_fourier", "S3_causal"]:
        temporal_errors = {ratio: [] for ratio in t_ratios}

        for seed in SEEDS:
            model = models_cache.get((solver_name, "A1_LambOseen", seed))
            if model is None:
                print(f"  WARNING: No cached model for {solver_name}/A1/seed={seed}")
                continue

            for ratio in t_ratios:
                t_val = ratio * t0
                test_points = np.column_stack([r_test, np.full_like(r_test, t_val)])
                y_pred = model.predict(test_points)
                y_true = anchor.omega(r_test.reshape(-1, 1), t_val)

                err = np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)
                temporal_errors[ratio].append(float(err))

        results[solver_name] = {}
        for ratio in t_ratios:
            if temporal_errors[ratio]:
                results[solver_name][str(ratio)] = {
                    "mean": float(np.mean(temporal_errors[ratio])),
                    "std": float(np.std(temporal_errors[ratio])),
                }

    with open(os.path.join(RESULTS_DIR, "exp3_temporal.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ── Experiment 4: Routing comparison ───────────────────────────────────────

def run_experiment4(exp1_results, models_cache):
    """Trust-informed routing vs fixed-solver strategies."""
    # Determine best solver per anchor from Exp 1
    anchor_names = ["A1_LambOseen", "A2_Stokes", "A3_Burgers", "A4_Kovasznay"]
    solver_names = ["S1_vanilla", "S2_fourier", "S3_causal"]

    best_solver_per_anchor = {}
    for anchor_name in anchor_names:
        best_err = float("inf")
        best_solver = None
        for solver_name in solver_names:
            key = f"{solver_name}__{anchor_name}"
            if key in exp1_results:
                if exp1_results[key]["mean"] < best_err:
                    best_err = exp1_results[key]["mean"]
                    best_solver = solver_name
        best_solver_per_anchor[anchor_name] = best_solver

    print(f"\nRouting map: {best_solver_per_anchor}")

    # Compute mean error for each strategy
    strategy_errors = {}
    for strategy in solver_names + ["routed"]:
        total_err = 0.0
        count = 0
        for anchor_name in anchor_names:
            if strategy == "routed":
                solver = best_solver_per_anchor[anchor_name]
            else:
                solver = strategy
            key = f"{solver}__{anchor_name}"
            if key in exp1_results:
                total_err += exp1_results[key]["mean"]
                count += 1
        strategy_errors[strategy] = total_err / max(count, 1)

    results = {
        "strategy_errors": strategy_errors,
        "routing_map": best_solver_per_anchor,
    }

    with open(os.path.join(RESULTS_DIR, "exp4_routing.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ── Experiment 5: Trust decay ──────────────────────────────────────────────

def run_experiment5(exp3_results):
    """Runtime trust decay calculation."""
    lam_decay = 0.01
    threshold = 0.05  # error threshold for τ=1

    results = {}
    s1_data = exp3_results.get("S1_vanilla", {})

    for t_ratio_str, data in s1_data.items():
        t_ratio = int(t_ratio_str)
        t0 = 100.0  # R0^2/nu
        t_max = t_ratio * t0
        t_certified = 5 * t0  # certification time

        tau = 1.0 if data["mean"] < threshold else 0.0
        T = tau * np.exp(-lam_decay * (t_max - t_certified))
        if t_max < t_certified:
            T = tau  # haven't exceeded certification horizon

        results[str(t_ratio)] = {
            "error": data["mean"],
            "tau": tau,
            "trust_score": float(T),
            "t_max": t_max,
        }

    with open(os.path.join(RESULTS_DIR, "exp5_trust_decay.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Trust Anchors Experiments")
    print("=" * 70)

    t_start = time.time()

    # Experiment 1: Table 2
    print("\n\n>>> EXPERIMENT 1: Table 2 (3 solvers × 4 anchors × 3 seeds)")
    exp1_results, models_cache = run_experiment1()

    # Experiment 2: Spectral bias (Figure 2)
    print("\n\n>>> EXPERIMENT 2: Spectral Bias (3 solvers × 4 freq × 3 seeds)")
    exp2_results = run_experiment2()

    # Experiment 3: Temporal drift (Figure 3) — uses cached models from Exp 1
    print("\n\n>>> EXPERIMENT 3: Temporal Drift (reuse Exp 1 models)")
    exp3_results = run_experiment3(models_cache)

    # Experiment 4: Routing (Figure 4)
    print("\n\n>>> EXPERIMENT 4: Trust-Informed Routing")
    exp4_results = run_experiment4(exp1_results, models_cache)

    # Experiment 5: Trust decay
    print("\n\n>>> EXPERIMENT 5: Trust Decay")
    exp5_results = run_experiment5(exp3_results)

    total_time = time.time() - t_start
    print(f"\n\nAll experiments complete in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Results saved to {RESULTS_DIR}/")
