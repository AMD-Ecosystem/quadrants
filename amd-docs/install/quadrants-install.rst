.. meta::
  :description: installing Quadrants for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, Quadrants

.. _quadrants-on-rocm-installation:

********************************************************************
Quadrants on ROCm installation
********************************************************************

System requirements
====================================================================

To use Quadrants `1.3.3 <https://github.com/AMD-Ecosystem/quadrants/tree/rocm10.1.0_r26.10>`__, you need the following prerequisites:

- **ROCm version:** `10.1.0 <https://rocm.docs.amd.com/en/docs-10.1.0/>`__
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI300X, MI325X, MI350X and MI355X
- **Python:** `3.12 <https://www.python.org/downloads/release/python-31215/>`__

Install Quadrants
====================================================================

This branch builds Quadrants for AMD GPUs on ROCm 10.1.0. The image installs PyTorch
``2.13.0+rocm10.1.0`` and the Quadrants wheel compiled with AMDGPU on, and CUDA and Vulkan
off. It keeps the C++ test binary and this source tree so the suites can run. It does not
install Genesis. The base image is ``rocm/dev-ubuntu-24.04:10.1.0-full``. The host kernel
driver must match ROCm 10.1.

Build the Docker image and run the container
--------------------------------------------------------------------

Clone this branch and build from the repository root.

1. Clone the Quadrants repository:

   .. code-block:: bash

      git clone -b rocm10.1.0_r26.10 https://github.com/AMD-Ecosystem/quadrants.git
      cd quadrants

2. Build the Docker image:

   .. code-block:: bash

      docker build -t quadrants-release-rocm10.1 .

3. Run the container on the AMD GPU. On a machine that also has a display GPU, set all three
   device variables to the CDNA device. That is GPU 0 on the machine used for this release.
   The shell starts in ``/src/quadrants``.

   .. code-block:: bash

      docker run --rm -it \
        --device=/dev/kfd --device=/dev/dri \
        --group-add "$(stat -c '%g' /dev/kfd)" \
        -e HIP_VISIBLE_DEVICES=0 \
        -e ROCR_VISIBLE_DEVICES=0 \
        -e CUDA_VISIBLE_DEVICES=0 \
        quadrants-release-rocm10.1 \
        bash

   .. note::

      Do not set ``QD_AMDGPU_V520``.

Test Quadrants
====================================================================

Inside that shell, set the C++ runtime path once, then run the four suites in this order.
Importing Quadrants prints a banner, so ``QD_LIB_DIR`` is the last line of that command.

.. code-block:: bash

   export QD_LIB_DIR=$(python -c 'import os, quadrants as qd; print(os.path.join(qd.__path__[0], "_lib", "runtime"))' | tail -n 1)

.. code-block:: bash

   /opt/quadrants/quadrants_cpp_tests --gtest_filter='AMDGPU.*'
   /opt/quadrants/quadrants_cpp_tests --gtest_filter='-AMDGPU.*'
   python tests/run_tests.py -t 4 -r 1 -v --arch amdgpu -m "not needs_torch"
   python tests/run_tests.py -t 4 -r 1 -v --arch amdgpu -m "needs_torch"

The first command is the C++ suite on the AMD GPU. The second is the C++ suite that does not
use the AMD GPU. The third is the Python suite without torch. The fourth is the Python suite
that needs torch.

Example Quadrants kernel
====================================================================

Paste this into the shell of the container. It compiles one kernel on the AMD GPU, adds 1 to
each element of an array, and checks the result.

.. code-block:: bash

   python << 'EOF'
   import numpy as np
   import quadrants as qd
   qd.init(arch=qd.amdgpu)
   n = 8
   @qd.kernel
   def add_one(x: qd.types.ndarray(qd.i32, ndim=1)):
       for i in range(x.shape[0]):
           x[i] = x[i] + 1
   values = qd.ndarray(qd.i32, shape=(n,))
   values.from_numpy(np.zeros(n, dtype=np.int32))
   add_one(values)
   result = values.to_numpy()
   print(result)
   assert np.array_equal(result, np.ones(n, dtype=np.int32))
   print("Quadrants kernel passed")
   EOF

A passing run prints eight 1s and then ``Quadrants kernel passed``.
