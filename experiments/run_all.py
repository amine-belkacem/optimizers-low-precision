"""Regenerate every result in results/. Run from the repository root:

    python -m experiments.run_all
"""

from experiments import differential, rosenbrock

if __name__ == "__main__":
    for experiment in (differential, rosenbrock):
        print(f"--- {experiment.__name__}")
        experiment.main()
