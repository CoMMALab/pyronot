#!/usr/bin/env bash
# Build _fk_cuda.so from _fk_cuda_kernel.cu.
#
# Usage (from repo root):
#   bash src/pyronot/cuda_kernels/build_fk_cuda.sh
#
# Requirements:
#   - nvcc (CUDA toolkit)
#   - jaxlib >= 0.4.14 installed in the active Python environment
#     (provides the xla/ffi/api/ffi.h headers)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/_fk_cuda_kernel.cu"
OUT="${SCRIPT_DIR}/_fk_cuda_lib.so"

# Locate the jaxlib include directory that ships xla/ffi/api/ffi.h.
JAXLIB_INC="$(python3 -c \
  "import os, jaxlib; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))")"

if [ ! -f "${JAXLIB_INC}/xla/ffi/api/ffi.h" ]; then
  echo "ERROR: xla/ffi/api/ffi.h not found under ${JAXLIB_INC}"
  echo "Make sure jaxlib >= 0.4.14 is installed in your Python environment."
  exit 1
fi

# GPU architecture flag.
# -arch=native (CUDA 11.6+) targets the installed GPU automatically.
# Override for a specific arch: GPU_ARCH=-arch=sm_80 bash build_fk_cuda.sh
GPU_ARCH="${GPU_ARCH:--arch=native}"

nvcc \
  -O3 \
  -std=c++17 \
  --use_fast_math \
  ${GPU_ARCH} \
  --shared \
  --compiler-options "-fPIC" \
  -I"${JAXLIB_INC}" \
  -o "${OUT}" \
  "${SRC}"

echo "Built: ${OUT}"
