#!/usr/bin/env bash
# Milestone J5 on a Raspberry Pi 5 (or any Linux machine): set up and run the
# same step benchmark as on the PC.
#
#   git clone https://github.com/amine-belkacem/optimizers-low-precision.git
#   cd optimizers-low-precision
#   bash bench/run_pi_benchmark.sh
#
# Nothing is installed system-wide: uv goes into .tools/, Python 3.12 and
# NumPy (same version as on the PC) into .venv-bench/. No PyTorch: the
# benchmark only needs NumPy. About 5 minutes on a Pi 5. Do not use the Pi
# while it runs, and keep its cooler on: the script records the temperature
# and whether the CPU was throttled.
set -euo pipefail

TAG="${1:-pi5}"
UV_VERSION="0.12.21"
PYTHON_VERSION="3.12"

cd "$(dirname "$0")/.."

for tool in git python3; do
    command -v "$tool" >/dev/null || { echo "missing: $tool (sudo apt install python3-venv git)"; exit 1; }
done

echo "== 1/3 Python $PYTHON_VERSION environment"
if [ ! -x .tools/bin/uv ]; then
    python3 -m venv .tools
    .tools/bin/pip install --quiet "uv==$UV_VERSION"
fi
[ -x .venv-bench/bin/python ] || .tools/bin/uv venv --python "$PYTHON_VERSION" .venv-bench
.tools/bin/uv pip install --quiet --python .venv-bench/bin/python -r bench/requirements-pi.txt
PY=.venv-bench/bin/python
export PYTHONPATH="src:."

echo "== 2/3 Benchmark tests on this machine"
$PY -m pytest -q -p no:cacheprovider tests/test_bench.py

echo "== 3/3 Benchmark (about 5 minutes)"
governor=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
[ -r "$governor" ] && echo "cpu governor: $(cat "$governor")"
command -v vcgencmd >/dev/null && echo "before: $(vcgencmd measure_temp), $(vcgencmd get_throttled)"
$PY -m bench.step_bench --tag "$TAG"
command -v vcgencmd >/dev/null && echo "after:  $(vcgencmd measure_temp), $(vcgencmd get_throttled)"

echo
echo "Done. Copy these two files back into results/ on the PC:"
echo "  results/bench_${TAG}.csv"
echo "  results/bench_${TAG}_machine.json"
