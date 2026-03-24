#!/usr/bin/env bash
# Build _chomp_trajopt_cuda_lib.so from _chomp_trajopt_cuda_kernel.cu.
#
# Usage (from repo root):
#   bash src/pyronot/cuda_kernels/build_chomp_trajopt_cuda.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/_chomp_trajopt_cuda_kernel.cu"
OUT="${SCRIPT_DIR}/_chomp_trajopt_cuda_lib.so"

JAXLIB_INC="$(python3 -c \
  "import os, jaxlib; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))")"

if [ ! -f "${JAXLIB_INC}/xla/ffi/api/ffi.h" ]; then
  echo "ERROR: xla/ffi/api/ffi.h not found under ${JAXLIB_INC}"
  echo "Make sure jaxlib >= 0.4.14 is installed in your Python environment."
  exit 1
fi

GPU_ARCH="${GPU_ARCH:--arch=native}"

nvcc \
  -O3 \
  -std=c++17 \
  ${GPU_ARCH} \
  --shared \
  --compiler-options "-fPIC" \
  -I"${JAXLIB_INC}" \
  -o "${OUT}" \
  "${SRC}"

echo "Built: ${OUT}"
