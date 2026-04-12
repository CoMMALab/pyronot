#!/usr/bin/env bash
# Build all CUDA kernels.
#
# Usage (from repo root):
#   bash src/pyronot/cuda_kernels/build_all.sh
#   bash src/pyronot/cuda_kernels/build_all.sh --debug
#
# Override GPU arch for all kernels:
#   GPU_ARCH=-arch=sm_80 bash src/pyronot/cuda_kernels/build_all.sh

set -euo pipefail

DEBUG_FLAG=""
MAX_JOINTS_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      DEBUG_FLAG="--debug"
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

BUILD_ARGS=()
if [[ -n "${DEBUG_FLAG}" ]]; then
  BUILD_ARGS+=("${DEBUG_FLAG}")
fi
if [[ -n "${MAX_JOINTS_OVERRIDE}" ]]; then
  if ! [[ "${MAX_JOINTS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --max-joints must be a positive integer, got '${MAX_JOINTS_OVERRIDE}'"
    exit 1
  fi
  BUILD_ARGS+=("--max-joints" "${MAX_JOINTS_OVERRIDE}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/build_fk_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_collision_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_hjcd_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_ls_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_brownian_motion_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_hit_and_run_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_svgd_region_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_mppi_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_sqp_ik_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_sco_trajopt_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_ls_trajopt_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_chomp_trajopt_cuda.sh" "${BUILD_ARGS[@]}"
bash "${SCRIPT_DIR}/build_stomp_trajopt_cuda.sh" "${BUILD_ARGS[@]}"

echo "All CUDA kernels built successfully."
