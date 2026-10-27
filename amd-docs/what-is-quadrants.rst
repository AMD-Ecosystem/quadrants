.. meta::
  :description: What is Quadrants?
  :keywords: Quadrants, documentation, compiler, simulation, physics, framework, GPU, AMD, ROCm, overview, introduction

.. _what-is-quadrants:

********************************************************************
What is Quadrants?
********************************************************************

Quadrants is a multi-platform compiler for physics simulation. You write kernels in
Python. Quadrants lowers them to parallel code for AMD ROCm (HIP),
Vulkan (SPIR-V), Apple Metal, and x86 and ARM64 CPUs. Genesis uses it for the numeric
kernels inside its solvers, collision code, and linear algebra.

The project was forked from `Taichi Lang <https://rocm.docs.amd.com/projects/taichi/en/latest/what-is-taichi.html>`__
and is now an independent compiler. It does not aim to stay compatible with upstream Taichi Lang,
as development is led by Genesis AI. This ROCm edition is the
`rocm10.1.0_r26.10 <https://github.com/AMD-Ecosystem/quadrants/tree/rocm10.1.0_r26.10>`__
branch of `AMD-Ecosystem/quadrants <https://github.com/AMD-Ecosystem/quadrants>`__.
The user guide is at `genesis-embodied-ai.github.io/quadrants <https://genesis-embodied-ai.github.io/quadrants/index.html>`__.

Why Quadrants?
====================================================================

Quadrants exists so a physics engine can keep one kernel language and still run on the
machines researchers actually have:

- **One kernel source, several backends:** A function marked ``@qd.kernel`` or ``@qd.func``
  is the source of the AMDGPU, Metal, Vulkan, and CPU programs. Switching backends
  does not mean rewriting the solver.

- **Built for large batched simulation:** The compiler's layout types, cache, and GPU graphs
  are aimed at the launch and memory patterns of a robotics simulator, not at a general
  graphics sandbox.

- **Room to extend the math:** Autodiff, subgroup operations, and register-resident tiles
  are available to kernel authors, which is how a rigid solver can implement its own linear
  algebra instead of calling a separate library for every step.

- **A small surface:** GUI, mobile, and several older graphics backends were removed so the
  compiler that Genesis ships is the one that is still maintained.

Features and use cases
====================================================================

Quadrants provides the following features:

- **Python kernel language:** ``qd.init`` selects the architecture. Fields, ndarrays, and
  ``qd.Tensor`` hold simulation state. Dataclass structs pass through ``qd.func`` with no
  extra kernel-runtime cost. ``BufferView`` is a bounds-checked window onto an ndarray.

- **AMDGPU, Metal, Vulkan, and CPU backends:** The ROCm 10.1 image enables AMDGPU and
  leaves CUDA and Vulkan off. The same source can target the other architectures in a build
  that turns them on.

- **Launch and compile speed:** Fastcache skips front-end parsing when a kernel has already
  been compiled. GPU graphs capture a sequence of kernels so a step is one launch.
  ``perf_dispatch`` times alternate implementations and keeps the faster one.

- **SIMT building blocks:** Subgroup shuffle and reductions cover AMDGPU, Metal, and
  Vulkan. Tile types keep 16x16 and 32x32 matrices in registers for blocked factorizations.

- **Autodiff:** Reverse-mode differentiation, including dynamic loops, plus forward mode,
  custom gradients, and a tape.

- **PyTorch interop:** ``to_torch(copy=False)`` and ``to_numpy(copy=False)`` use DLPack on
  CPU, AMDGPU, and Metal, and a torch tensor can be passed into a kernel.

- **A Python interpreter for kernels:** ``qd.init(qd.python)`` runs a kernel as ordinary
  Python so it can be stepped through in a debugger.

Quadrants is commonly used in the following scenarios:

- **As the Genesis compiler:** Rigid, deformable, and particle solvers in Genesis are
  Quadrants kernels. The ROCm Genesis image is a Quadrants wheel with Genesis installed
  on top.

- **Standalone kernel development:** ``pip install quadrants``, or the ROCm image
  ``quadrants-release-rocm10.1``, is enough to compile and run kernels without Genesis.

- **Portable physics code:** A prototype written once can be checked on a CPU, then on an
  AMD GPU, by changing the architecture passed to ``qd.init``.

- **Differentiable simulation:** Reverse-mode autodiff through a kernel is how a caller gets
  gradients without hand-writing an adjoint.
