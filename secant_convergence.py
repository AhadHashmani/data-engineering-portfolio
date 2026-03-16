"""
Secant Method with Convergence Rate Analysis
=============================================
Finds roots of polynomials using the secant method and computes
empirical convergence rates (linear and superlinear).

Author: Ahad Hashmani
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt


class STATE(Enum):
    SUCCESS = 0
    WONT_STOP = 1
    BAD_ITERATE = 2


@dataclass
class SecantResult:
    state: STATE
    root: float
    iterates: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    linear_rates: List[Optional[float]] = field(default_factory=list)
    superlinear_rates: List[Optional[float]] = field(default_factory=list)
    iterations: int = 0


def f(x: float) -> float:
    """Target polynomial: 54x^6 + 45x^5 - 102x^4 - 69x^3 + 35x^2 + 16x - 4"""
    return 54*x**6 + 45*x**5 - 102*x**4 - 69*x**3 + 35*x**2 + 16*x - 4


def secant(
    g0: float,
    g1: float,
    tol: float,
    max_iters: int,
    func: Callable[[float], float] = f,
) -> SecantResult:
    """
    Secant method with full error and convergence rate tracking.

    Parameters
    ----------
    g0, g1     : initial guesses
    tol        : convergence tolerance on |dx|
    max_iters  : maximum iterations before WONT_STOP
    func       : function whose root is sought

    Returns
    -------
    SecantResult dataclass containing root, iterates, errors, and rates
    """
    iterates = [g0, g1]
    x = g1
    f0 = func(g0)

    for _ in range(1, max_iters + 1):
        fx = func(x)
        if abs(fx - f0) < 1e-25:
            return SecantResult(state=STATE.BAD_ITERATE, root=x,
                                iterates=iterates, iterations=len(iterates))

        dx = -fx * (x - g0) / (fx - f0)
        g0, f0 = x, fx
        x += dx
        iterates.append(x)

        if abs(dx) <= tol:
            break
    else:
        true_r = iterates[-1]
        errors = [abs(v - true_r) for v in iterates]
        return SecantResult(state=STATE.WONT_STOP, root=x,
                            iterates=iterates, errors=errors,
                            iterations=len(iterates))

    true_r = iterates[-1]
    errors = [abs(v - true_r) for v in iterates]

    linear_rates: List[Optional[float]] = [None, None]
    superlinear_rates: List[Optional[float]] = [None, None]

    for i in range(2, len(iterates)):
        e_i = errors[i]
        e_prev = errors[i - 1]
        e_prev2 = errors[i - 2]

        lr = e_i / e_prev if e_prev > 1e-15 else None
        slr = e_i / (e_prev * e_prev2) if (e_prev * e_prev2) > 1e-15 else None

        linear_rates.append(lr)
        superlinear_rates.append(slr)

    return SecantResult(
        state=STATE.SUCCESS,
        root=true_r,
        iterates=iterates,
        errors=errors,
        linear_rates=linear_rates,
        superlinear_rates=superlinear_rates,
        iterations=len(iterates),
    )


def print_table(result: SecantResult, label: str) -> None:
    """Print a formatted convergence table for a single root."""
    print(f"\n{'='*70}")
    print(f"  {label}  |  Root: {result.root:.10f}  |  Status: {result.state.name}")
    print(f"{'='*70}")
    print(f"{'Iter':<5} | {'x_i':>14} | {'e_i':>12} | {'r_linear':>12} | {'r_superlinear':>14}")
    print(f"{'-'*5}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}")

    for i, (xi, ei) in enumerate(zip(result.iterates, result.errors)):
        rl = f"{result.linear_rates[i]:.6f}" if (
            i < len(result.linear_rates) and result.linear_rates[i] is not None
        ) else "N/A"
        rsl = f"{result.superlinear_rates[i]:.6f}" if (
            i < len(result.superlinear_rates) and result.superlinear_rates[i] is not None
        ) else "N/A"
        print(f"{i:<5} | {xi:>14.10f} | {ei:>12.3e} | {rl:>12} | {rsl:>14}")


def plot_convergence(results: List[Tuple[str, SecantResult]], save_path: str = "convergence.png") -> None:
    """Plot error decay on a log scale for all roots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, res in results:
        if res.state == STATE.SUCCESS and any(e > 0 for e in res.errors):
            axes[0].semilogy(res.errors, marker="o", markersize=3, label=label)

            valid_lr = [(i, r) for i, r in enumerate(res.linear_rates) if r is not None]
            if valid_lr:
                idx, rates = zip(*valid_lr)
                axes[1].plot(idx, rates, marker="s", markersize=3, label=label)

    axes[0].set_title("Error Decay per Iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Absolute Error (log scale)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)

    axes[1].set_title("Linear Convergence Rate e_{i}/e_{i-1}")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Rate")
    axes[1].axhline(y=1.0, color="red", linestyle="--", alpha=0.4, label="Rate = 1")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nConvergence plot saved to {save_path}")


def main() -> None:
    initial_guesses: List[Tuple[float, float]] = [
        (-1.5, -1.3),
        (-0.8, -0.6),
        (0.1,  0.3),
        (0.4,  0.6),
        (1.0,  1.2),
    ]

    TOL = 1e-10
    MAX_ITERS = 100
    labeled_results = []

    for idx, (x0, x1) in enumerate(initial_guesses, start=1):
        label = f"Root {idx} (x0={x0}, x1={x1})"
        result = secant(x0, x1, TOL, MAX_ITERS)
        print_table(result, label)
        labeled_results.append((f"Root {idx}", result))

    print("\n" + "="*50)
    print("Final Root Summary")
    print("="*50)
    for label, res in labeled_results:
        status = res.state.name
        print(f"{label}: {res.root:.10f}  [{status}]  ({res.iterations} iters)")

    plot_convergence(labeled_results)


if __name__ == "__main__":
    main()
