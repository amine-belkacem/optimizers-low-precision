"""Regenerate every result in results/. Run from the repository root:

    python -m experiments.run_all

About 4 to 5 minutes on a laptop CPU, most of it in the learning-rate sweep.
"""

from experiments import differential, lr_sweep, rosenbrock

if __name__ == "__main__":
    for experiment in (differential, rosenbrock, lr_sweep):
        print(f"--- {experiment.__name__}")
        experiment.main()
