#!/usr/bin/env bash
# Build _hjcd_ik_cuda_lib.so from _hjcd_ik_cuda_kernel.cu.
#
# Usage (from repo root):
#   bash src/pyroffi/cuda_kernels/build_hjcd_ik_cuda.sh
#   bash src/pyroffi/cuda_kernels/build_hjcd_ik_cuda.sh --debug
#
# Requirements:
#   - nvcc (CUDA toolkit)
#   - jaxlib >= 0.4.14 installed in the active Python environment
#     (provides the xla/ffi/api/ffi.h headers)

set -euo pipefail

DEBUG=0
MAX_JOINTS_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      DEBUG=1
      shift
      ;;
    --max-joints)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --max-joints requires an integer value"
        exit 1
      fi
      MAX_JOINTS_OVERRIDE="$2"
      shift 2
      ;;
    --max-joints=*)
      MAX_JOINTS_OVERRIDE="${1#*=}"
      shift
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      exit 1
      ;;
  esac
done

MAX_JOINTS_FLAG=""
if [[ -n "${MAX_JOINTS_OVERRIDE}" ]]; then
  if ! [[ "${MAX_JOINTS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --max-joints must be a positive integer, got '${MAX_JOINTS_OVERRIDE}'"
    exit 1
  fi
  MAX_JOINTS_FLAG="-DMAX_JOINTS=${MAX_JOINTS_OVERRIDE}"
  echo "Overriding MAX_JOINTS=${MAX_JOINTS_OVERRIDE}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/_hjcd_ik_cuda_kernel.cu"
OUT="${SCRIPT_DIR}/_hjcd_ik_cuda_lib.so"

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
# Override for a specific arch: GPU_ARCH=-arch=sm_80 bash build_hjcd_ik_cuda.sh
GPU_ARCH="${GPU_ARCH:--arch=native}"

NVCC_OPT="-O3"
if [ "${DEBUG}" -eq 1 ]; then
  NVCC_OPT="-O0 -G -lineinfo"
  echo "Building in DEBUG mode (with -G for Nsight Compute)..."
fi

nvcc \
  ${NVCC_OPT} \
  -std=c++17 \
  ${MAX_JOINTS_FLAG} \
  ${GPU_ARCH} \
  --shared \
  --compiler-options "-fPIC" \
  -I"${SCRIPT_DIR}" \
  -I"${JAXLIB_INC}" \
  -o "${OUT}" \
  "${SRC}"

echo "Built: ${OUT}"
