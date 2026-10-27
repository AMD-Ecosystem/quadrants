.. meta::
  :description: Quadrants documentation
  :keywords: Quadrants, ROCm, documentation, compiler, simulation, physics, framework, GPU

.. _quadrants-documentation-index:

********************************************************************
Quadrants on ROCm documentation
********************************************************************

Use Quadrants on ROCm to compile Python physics kernels for AMD Instinct GPUs.
A kernel written once lowers to parallel code for the selected architecture.

Quadrants is a multi-platform compiler for physics simulation. You write kernels in
Python. Quadrants lowers them to parallel code for AMD ROCm (HIP),
Vulkan (SPIR-V), Apple Metal, and x86 and ARM64 CPUs. Genesis uses it for the numeric
kernels inside its solvers, collision code, and linear algebra.

The project was forked from `Taichi Lang <https://rocm.docs.amd.com/projects/taichi/en/latest/what-is-taichi.html>`__
and is now an independent compiler. It does not aim to stay compatible with upstream Taichi Lang,
as development is led by Genesis AI. This ROCm edition is the
`rocm10.1.0_r26.10 <https://github.com/AMD-Ecosystem/quadrants/tree/rocm10.1.0_r26.10>`__
branch of `AMD-Ecosystem/quadrants <https://github.com/AMD-Ecosystem/quadrants>`__.

The Quadrants public repository is located at
`https://github.com/AMD-Ecosystem/quadrants <https://github.com/AMD-Ecosystem/quadrants>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install Quadrants <install/quadrants-install>`

  .. grid-item-card:: Reference

      * `Quadrants user guide (upstream) <https://genesis-embodied-ai.github.io/quadrants/index.html>`__

To contribute to the documentation, refer to
`Contributing to Quadrants <https://github.com/AMD-Ecosystem/quadrants/blob/docs/26.10/docs/source/user_guide/contributing.md>`__.

You can find licensing information on the :doc:`Licensing <about/license>` page.
