#!/usr/bin/env bash
# Build all CUDA kernels.
#
# Usage (from repo root):
#   bash src/pyronot/cuda_kernels/build_all.sh
#
# Override GPU arch for all kernels:
#   GPU_ARCH=-arch=sm_80 bash src/pyronot/cuda_kernels/build_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/build_fk_cuda.sh"
bash "${SCRIPT_DIR}/build_collision_cuda.sh"
bash "${SCRIPT_DIR}/build_hjcd_ik_cuda.sh"
bash "${SCRIPT_DIR}/build_ls_ik_cuda.sh"
bash "${SCRIPT_DIR}/build_mppi_ik_cuda.sh"
bash "${SCRIPT_DIR}/build_sqp_ik_cuda.sh"

echo "All CUDA kernels built successfully."
