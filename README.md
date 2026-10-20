# low-precision-optimizers

I am writing SGD, Momentum, Adam and AdamW from scratch in NumPy, checking them
against `torch.optim`, and then looking at what happens to each of them when
the arithmetic drops to fp16, bfloat16 or fixed point, the way it does when you
try to train on a microcontroller, an accelerator or an FPGA.

> Work in progress. This repository is built in milestones and this README
> only describes what is already in it. No result is reported here until it
> has been measured.

## Status

| Milestone | Content | State |
|---|---|---|
| J0 | Repository skeleton, CI, figure style | done |
| J1 | The four optimizers in fp64, differential tests against PyTorch | planned |
| J2 | NumPy MLP on `digits`, learning-rate sensitivity | planned |
| J3 | Number format emulator (fp32, fp16, bf16, Qm.n) | planned |
| J4 | Low-precision study | planned |
| J5 | Step time and optimizer memory on a Raspberry Pi 5 | planned |

## Setup

Python 3.10 or newer. PyTorch is only used as the reference in the tests, so
the CPU wheel is enough:

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest
```

On Windows, replace `.venv/bin/` with `.venv\Scripts\`.

## Figures

Every figure in this repository is produced by a script under `figures/`, using
the shared style in `src/lpopt/plotting.py`. Nothing is drawn by hand.

## License

MIT, see [LICENSE](LICENSE).
