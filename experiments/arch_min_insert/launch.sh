#!/usr/bin/env bash
set -euo pipefail
ARCH_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
export PYTHONPATH="$ARCH_ROOT/multi-robot-assembly/multi_arm_assembly:$ARCH_ROOT/diffusion_policy_v1${PYTHONPATH:+:$PYTHONPATH}"
cd "$ARCH_ROOT/multi-robot-assembly/multi_arm_assembly"
exec "$ARCH_ROOT/.venv-arch/bin/python" "$ARCH_ROOT/experiments/arch_min_insert/run.py" "$@"
