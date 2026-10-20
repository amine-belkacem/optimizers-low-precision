"""Regenerate every result in results/. Run from the repository root:

    python -m experiments.run_all

About 15 minutes on an 8-core laptop: 5 for the float64 learning-rate sweep,
5 for the quadratic, 5 for the 600 low-precision runs (spread over the cores).
The low-precision study reads the learning rates chosen by the sweep, so the
order below matters.
"""

from experiments import differential, lr_sweep, precision_matrix, precision_summary, quadratic_stall, rosenbrock

if __name__ == "__main__":
    for experiment in (differential, rosenbrock, lr_sweep, quadratic_stall, precision_matrix, precision_summary):
        print(f"--- {experiment.__name__}")
        experiment.main()
