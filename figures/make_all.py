"""Redraw every figure in docs/img from results/. Run from the repository root:

    python -m figures.make_all
"""

from figures import differential, digits, formats, rosenbrock

if __name__ == "__main__":
    for figure in (differential, rosenbrock, digits, formats):
        figure.main()
        print(f"done: {figure.__name__}")
