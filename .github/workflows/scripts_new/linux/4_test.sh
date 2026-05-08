#!/bin/bash

set -ex

pip install --group test
pip install -r requirements_test_xdist.txt
export QD_LIB_DIR="$(python -c 'import quadrants as ti; print(ti.__path__[0])' | tail -n 1)/_lib/runtime"
./build/quadrants_cpp_tests

# NOTE(amdgpu): in this fork, 4_test.sh runs on the AMDGPU CI runner via
# linux.yml, so we target -a amdgpu. The upstream phased coverage flow is
# kept in 4_test_cuda.sh for the CUDA runner; mirror its TEST_EXIT pattern
# here so a single failure doesn't mask later phase results.
TEST_EXIT=0

# Phase 1: non-torch tests. The test_reduction_single* tests put the GPU
# under a lot of stress and can run out of time during the test run with
# lots of GPU workers; run them serially after the parallel block.
python tests/run_tests.py -v -r 3 -a amdgpu -t 16 -m "not needs_torch" -k "not test_reduction_single" || TEST_EXIT=$?
python tests/run_tests.py -v -r 3 -a amdgpu -t 16 -m "not needs_torch" -k "test_reduction_single" || TEST_EXIT=$?

# Phase 2: install torch (ROCm wheel) and run torch-dependent tests.
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
QD_KERNEL_COVERAGE=0 python tests/run_tests.py -v -r 3 -a amdgpu -t 16 -m needs_torch || TEST_EXIT=$?

exit $TEST_EXIT
