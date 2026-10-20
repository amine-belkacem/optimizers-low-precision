# optimizers-low-precision

[![tests](https://github.com/amine-belkacem/optimizers-low-precision/actions/workflows/tests.yml/badge.svg)](https://github.com/amine-belkacem/optimizers-low-precision/actions/workflows/tests.yml)

I am writing SGD, Momentum, Adam and AdamW from scratch in NumPy, checking them
against `torch.optim`, and then looking at what happens to each of them when
the arithmetic drops to fp16, bfloat16 or fixed point, the way it does when you
try to train on a microcontroller, an accelerator or an FPGA.

> Work in progress. This repository is built in milestones and this README
> only describes what is already in it. Every number below comes from a
> script in this repository and can be regenerated.

## Status

| Milestone | Content | State |
|---|---|---|
| J0 | Repository skeleton, CI, figure style | done |
| J1 | The four optimizers, differential tests against PyTorch, Rosenbrock | done |
| J2 | NumPy MLP on `digits`, learning-rate sensitivity over 10 seeds | done |
| J3 | Number format emulator (fp32, fp16, bf16, Qm.n) | done |
| J4 | Low-precision study | next |
| J5 | Step time and optimizer memory on a Raspberry Pi 5 | planned |

## The optimizers

All four live in [`src/lpopt/optimizers.py`](src/lpopt/optimizers.py), about
200 lines with the docstrings. They follow the update rules of `torch.optim`
(PyTorch 2.x), including two details that are easy to get wrong:

- **SGD with momentum** stores the raw gradient as its first buffer, without
  dampening. The textbook version `b1 = (1 - dampening) * g1` gives a
  different first step as soon as dampening is not zero.
- **Adam vs AdamW.** Adam's `weight_decay` is an L2 term added to the gradient,
  and Adam then divides it by `sqrt(v)` like the rest of the gradient. On the
  first step it is almost completely normalized away: with `p = 1`, `g = 0.5`,
  `lr = 0.1` and a decay of 0.5, Adam + L2 moves `p` to 0.900000001 and
  AdamW to 0.850000002. AdamW shrinks the parameters directly,
  `p *= 1 - lr * weight_decay`, before the Adam step. Both values are checked
  by hand in [`tests/test_optimizers_hand.py`](tests/test_optimizers_hand.py).

The parameters are updated in place and the arithmetic stays in their dtype,
so the same code runs in float64 or float32. That matters for the next
milestones, where the format becomes the variable under study.

## How I check them

| Test | What it does |
|---|---|
| Hand-computed steps | 1 or 2 steps of each optimizer on a scalar. Expected values derived by hand, then confirmed with exact fractions and 40-digit decimals |
| Differential, 1000 steps | Same initial parameters and same gradients fed to lpopt and `torch.optim`, 9 configurations, float64 and float32 |
| Differential, random settings | Hypothesis draws 60 random configurations (learning rate, betas, eps, momentum, dampening, Nesterov, weight decay) |
| Mutation testing | 10 plausible bugs injected into a copy of `optimizers.py`. The differential bench has to catch every one |
| Closed form | Plain SGD on a quadratic follows `x_k = (1 - lr*a)^k x_0`. With `a = 1, lr = 0.5` this is exact, so the test compares bit for bit |

The differential tests feed the same precomputed gradients to both sides
(open loop), so any difference comes from the optimizers and not from a
feedback loop that would amplify it.

### Distance to PyTorch

![Normwise relative error between lpopt and torch.optim](docs/img/differential_error.png)

Worst case over 10 seeds and 1000 steps, default settings of each optimizer
([`results/differential_summary.csv`](results/differential_summary.csv)):

| Optimizer | float64 | float32 |
|---|---|---|
| SGD | 9.0e-16 | 6.5e-7 |
| Momentum | 6.4e-16 | 2.7e-7 |
| Adam | 4.3e-16 | 2.5e-7 |
| AdamW | 4.6e-16 | 3.6e-7 |

That is a handful of roundings in each dtype, against test thresholds of
1e-10 (float64) and 1e-5 (float32). The threshold has margin, but it is not
lax: the hardest mutant to catch, `eps` moved inside the square root, still
produces an error of 9.5e-4.

### Why the error is not zero

My first run in float32 gave **exactly zero** error for SGD while float64 did
not, which is backwards. Comparing torch against exact rational arithmetic
explained it: on my CPU (AVX2, PyTorch 2.14.1), `p.add_(g, alpha=-lr)` uses a
fused multiply-add, one rounding instead of two, but only on full vector
blocks of 8 float64 or 16 float32 values. The tail of the tensor goes
through the plain multiply then add. My first test tensors had 15 and 7
elements, so in float32 they never reached a full block. The bench now also
has a 64-element parameter so both code paths run in both dtypes.

So the same element can be rounded differently by torch depending on the size
of the tensor it sits in. lpopt rounds after every operation, which is
predictable and is what the format emulator of J3 will build on.

## Rosenbrock

`f(x, y) = (1 - x)^2 + 100 (y - x^2)^2`, from the classic start (-1.2, 1),
5000 steps. Comparing optimizers at one hand-picked learning rate mostly
measures how well that rate was picked, so each optimizer gets the same
11-point grid from 1e-5 to 1 and keeps its best rate.

![Paths of the four optimizers on Rosenbrock](docs/img/rosenbrock_paths.png)

![Loss along the steps](docs/img/rosenbrock_loss.png)

![Final loss against learning rate](docs/img/rosenbrock_lr_grid.png)

What I take from it ([`results/rosenbrock_lr_grid.csv`](results/rosenbrock_lr_grid.csv)):

- **SGD** cannot go faster: at 1e-2 it diverges, at its best rate (1e-3) it
  is still crawling along the valley, loss 3.8e-3 after 5000 steps.
- **Momentum** reaches 1.2e-30 at 3.2e-3, 3.5e-16 away from (1, 1). That is
  1.6 times the spacing between float64 numbers next to 1.0 (2.2e-16): it has
  reached the float64 floor and oscillates there. This is the kind of limit
  the low-precision milestones are about.
- **Adam** has the widest working range (it never diverged on this grid) and
  reaches 4.4e-31 at 3.2e-2, but takes about 4500 steps to get there.
- **AdamW** stalls around 8.7e-5, 2.1e-2 away from the minimum. This is not a
  bug, the decay does it: at (1, 1) the gradient is zero, so the Adam step
  is zero, but the decay still pulls the parameters towards the origin.
  (1, 1) is no longer a fixed point.

Limits: one start point, a 2-parameter problem, and a deterministic run, so
there is nothing to average. Rosenbrock shows how each update rule moves.
It does not say which optimizer trains a network best, that is what J2 is
for.

## A real network: MLP on digits

**Setup.** A 64-32-10 tanh MLP with a softmax cross-entropy, forward and
backward written by hand in [`src/lpopt/mlp.py`](src/lpopt/mlp.py). The
gradient is checked against central differences (error < 1e-8) and against
`torch.autograd` (relative error < 1e-12). The data is `digits`: 1797 8x8
images of handwritten digits (UCI, CC BY 4.0, shipped with scikit-learn).
It is split once, stratified: 1078 train, 359 validation, 360 test.

**Protocol.** Mini-batches of 64, 50 epochs, 16 learning rates from 1e-4 to
10 (a third of a decade apart), 10 seeds per point. A seed changes the
initial weights and the batch order, never the split. Each optimizer's rate
is chosen on the mean validation loss at the last epoch. The test set is
only read at that rate. In total, 640 training runs, about 3 minutes on a
laptop CPU.

I had planned a grid up to 1. A first look with one seed put SGD's best
rate exactly at 1, on the edge, and a best value on the edge of a grid says
nothing about what lies past it. So the grid goes to 10.

![Learning-rate sensitivity on digits](docs/img/lr_sensitivity_digits.png)

| Optimizer | Chosen lr | Test accuracy (10 seeds) | Test loss | Usable lr range* |
|---|---|---|---|---|
| SGD | 1.0 | 97.11 % ± 0.35 | 0.087 ± 0.010 | 0.22 to 1.0 (0.67 decade) |
| Momentum 0.9 | 0.22 | 97.53 % ± 0.38 | 0.090 ± 0.014 | 0.022 to 0.46 (1.33 decades) |
| Adam | 0.022 | 97.28 % ± 0.39 | 0.091 ± 0.012 | 0.0046 to 0.046 (1.0 decade) |
| AdamW, decay 0.01 | 0.022 | 97.31 % ± 0.37 | 0.087 ± 0.010 | 0.0046 to 0.046 (1.0 decade) |

\* rates whose mean validation accuracy is within one percentage point of
the optimizer's best rate. I fixed this rule before looking at which
optimizer it favors. Source: [`results/lr_sweep_digits_best.csv`](results/lr_sweep_digits_best.csv).

What the numbers say, and what they do not:

- **At its best rate, no optimizer wins.** The four test accuracies sit
  within 0.42 points, about one standard deviation. With 360 test images,
  one image is 0.28 points. On this problem, ranking the optimizers by
  their best score would be reading noise.
- **The difference is in how much tuning they need.** Momentum keeps working
  over 1.33 decades of learning rate, Adam and AdamW over 1.0, plain SGD over
  0.67. Adam having a narrower window than Momentum goes against what is
  often said about Adam. That holds for this network, this criterion and 50
  epochs, nothing more.
- **Momentum 0.9 at lr behaves like SGD at 10·lr**, the classic effective
  rate `lr / (1 - momentum)`. Up to an effective rate of 1 the two curves
  lie on top of each other, within the seed spread. Past that point the
  equivalence breaks: at an effective rate of 4.6, Momentum still reaches
  98.4 % on validation while SGD has collapsed to chance.

![Momentum against SGD at the same effective learning rate](docs/img/momentum_effective_lr.png)

- **Lower training loss does not mean a better model.** At their chosen
  rates, Adam ends with a training loss of 1.2e-3 and SGD with 9.8e-3, 8
  times higher, yet their validation losses are 0.059 and 0.063. Most of
  what Adam gains on the training set does not carry over.
- **Adam and AdamW are almost the same here.** Per step the decay removes
  `lr x 0.01` of each weight, 2e-4 at the chosen rate. Over 850 steps it
  does change the weights, but the test results stay inside the noise.
- **Too large a rate does not blow up, it collapses.** No run produced an
  infinite or NaN loss. Past the window, accuracy falls to 10 %, the chance
  level, with a finite loss. I suspect the tanh units saturate, but I have
  not checked it.

![Training curves at the chosen learning rates](docs/img/training_curves_digits.png)

Limits: one small dataset, one architecture, a fixed budget of 50 epochs
with no learning-rate schedule, and the rate is picked on the last epoch
instead of the best one. The ranking of the usable windows could change
with any of these. This is a reference point in float64 for the
low-precision study, not a general benchmark of optimizers.

## Emulating number formats

Before training in fp16 or bf16 I need a way to compute in those formats
that I fully control. [`src/lpopt/numerics.py`](src/lpopt/numerics.py)
emulates fp32, fp16, bf16 and signed fixed point Qm.n (here Q7.8 and
Q15.16), with round-to-nearest-even or stochastic rounding.

The rule: to compute `a + b` in fp16, compute it in float64 and round the
result to fp16. This is exact, not an approximation. For +, -, *, / and
sqrt, rounding twice (exact value to float64, then float64 to the target)
gives the same answer as rounding once, provided float64 has at least
2p + 2 bits of precision, where p is the target's (Figueroa, 1995). fp32
has p = 24, fp16 11, bf16 8, so all three qualify. The format class refuses
anything more precise.

How I check it:

| Test | Result |
|---|---|
| Conversion of 1,000,000 random float64 values (1e-45 to 1e42) to fp16 and fp32 | bit-identical to NumPy |
| Same values, as float32, to bf16 | bit-identical to PyTorch |
| `+ - * / sqrt` on about 1,000,000 random operand pairs per operation and format (random bit patterns, so subnormals and extremes included) | bit-identical to NumPy float32, NumPy float16 and PyTorch bfloat16 |
| Hand values: ties to even, overflow at 65520 in fp16, sign of zero, saturation in Q7.8 | pass |
| Stochastic rounding on 200,000 draws | unbiased within 5 standard deviations |

One finding along the way: **PyTorch converts float64 to bfloat16 through
float32, so it rounds twice.** Take x = 1 + 2^-8 + 2^-30. It sits just above
the midpoint between the bf16 neighbors 1 and 1 + 2^-7, so the right answer
is 1.0078125. The float32 step rounds x onto the midpoint exactly, and the
tie then goes to the even neighbor: torch returns 1.0. On my million random
values this happened 10 times, and exact rational arithmetic says the
emulator was the nearest every time. This is why the bf16 check against
torch uses float32 inputs.

### Two predictions for the low-precision study

![Spacing of representable numbers in each format](docs/img/format_spacing.png)

The figure is computed from the format definitions, not from training.
It already shows where things should go wrong:

- **H1, Adam's eps disappears in fp16.** The smallest fp16 number is
  2^-24 ≈ 6e-8, and anything below half of it rounds to 0. Adam's default
  `eps = 1e-8` is below that, so it rounds to 0. In bf16, which keeps
  fp32's exponent range, it survives with 8 bits of precision. Both are
  tested.
- **H2, Adam's second moment cannot decay in bf16.** Each step multiplies
  `v` by beta2 = 0.999, a change of 0.1 %. bf16 values are 2^-8 to 2^-7
  apart in relative terms, so 0.1 % is always less than half a spacing.
  I checked all 32,512 positive normal bf16 values: `round(0.999 * v)`
  gives back `v` every time. In fp16 the same product changes every normal
  value. What this does to training, given that Adam also adds
  `(1 - beta2) g^2` to `v` at each step, is what J4 measures.

## Reproduce

Python 3.10 or newer. PyTorch is only the reference in the tests and in the
differential experiment, so the CPU wheel is enough:

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/pytest                          # 184 tests, about 35 s
.venv/bin/python -m experiments.run_all   # results/, about 5 min
.venv/bin/python -m figures.make_all      # docs/img/
```

On Windows, replace `.venv/bin/` with `.venv\Scripts\`.

## Layout

```
src/lpopt/       optimizers, number formats, MLP, tasks, training loops, figure style
experiments/     scripts that write results/ (CSV summaries are committed)
figures/         scripts that turn results/ into docs/img/
tests/           pytest + Hypothesis, run by GitHub Actions on every push
```

## References

- D. P. Kingma, J. Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015.
- I. Loshchilov, F. Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019.
- S. A. Figueroa, *When is double rounding innocuous?*, ACM SIGNUM Newsletter 30(3), 1995.
- IEEE 754-2019, *Standard for Floating-Point Arithmetic*.
- E. Alpaydin, C. Kaynak, *Optical Recognition of Handwritten Digits*, UCI Machine Learning Repository, 1998, CC BY 4.0.
- X. Glorot, Y. Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010 (weight initialization).
- H. H. Rosenbrock, *An Automatic Method for Finding the Greatest or Least Value of a Function*, The Computer Journal, 1960.
- [`torch.optim` documentation](https://pytorch.org/docs/stable/optim.html), the reference implementation.

## License

MIT, see [LICENSE](LICENSE).
