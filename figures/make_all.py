"""Redraw every figure in docs/img from results/. Run from the repository root:

    python -m figures.make_all
"""

from figures import differential, digits, rosenbrock

if __name__ == "__main__":
    for figure in (differential, rosenbrock, digits):
        figure.main()
        print(f"done: {figure.__name__}")
