"""IR-level assertions for AMDGPU atomic codegen (PR #38 follow-up).

These tests are a tripwire for the codegen contract introduced in PR #38:
every atomic on AMDGPU should lower to an LLVM ``atomicrmw`` or
``cmpxchg`` carrying ``syncscope("agent")`` and ``monotonic`` ordering.
The C++/Python correctness tests in ``test_atomic_amdgpu.py`` only catch
*observable* regressions; if a future LLVM upgrade or codegen change
silently reverts the ordering / scope to ``seq_cst`` / system, the
program will still produce correct results but will pay the cache-flush
cost the PR was designed to eliminate. These tests catch that case.

Implementation follows the pattern used in ``test_fn_attrs.py``: spawn a
child process with ``print_kernel_llvm_ir=True``, run a minimal kernel
that exercises one atomic shape, then parse the dumped ``.ll`` file and
assert on the IR text.

Each test isolates a single atomic shape so that when the assertion
fails, the diagnostic points at exactly which lowering path regressed.

Post-Phase-3 status:
  * Integer / FP-add / FP-mul CAS paths emit ``syncscope("agent")
    monotonic`` (asserted strictly).
  * f32/f64 ``atomic_min``/``atomic_max`` now route through the
    agent-scope CAS path (Phase 3.1), so they are also asserted
    strictly. No more mixed-scope: ``atomic_add`` and ``atomic_min`` on
    the same f32 / f64 field both use agent+monotonic.
  * The CAS-loop's initial load is now an atomic monotonic load (Phase
    3.2), which prevents the optimizer from legally hoisting it out of
    the retry block on future LLVM upgrades.
"""

import os
import pathlib
import re
import subprocess
import sys

import quadrants as qd

from tests import test_utils

RET_SUCCESS = 42

# Regexes that match the relevant atomic instructions. We intentionally do
# not pin the exact destination type / addrspace because those depend on
# SNode layout and may change innocuously; we only assert the
# ordering+scope tokens that this PR is responsible for.
_AGENT_MONO_RMW_RE = re.compile(r"atomicrmw\s+[^\n]*syncscope\(\"agent\"\)\s+monotonic")
_AGENT_MONO_CMPXCHG_RE = re.compile(r"cmpxchg\s+[^\n]*syncscope\(\"agent\"\)\s+monotonic(?:\s+monotonic)?")
_ANY_RMW_RE = re.compile(r"atomicrmw\s")
_ANY_CMPXCHG_RE = re.compile(r"cmpxchg\s")
_SYSTEM_SEQCST_RMW_RE = re.compile(r"atomicrmw\s+[^\n]*\sseq_cst\s*(?:,|$)")
_SYSTEM_SEQCST_CMPXCHG_RE = re.compile(r"cmpxchg\s+[^\n]*\sseq_cst\s+seq_cst")


# ---------------------------------------------------------------------------
# Child-process scaffolding.
# ---------------------------------------------------------------------------


_RUNTIME_BITCODE_MODULE_ID = "ModuleID = 'runtime_bitcode'"


def _read_all_ll(dump_dir: pathlib.Path) -> str:
    """Concatenate every user-kernel IR dump in ``dump_dir`` into a
    single string for grepping. We don't care which task the atomic
    ended up in; we only care that the overall translation unit contains
    the expected ops.

    The JIT writes one ``quadrants_kernel_amdgpu_llvm_ir_*.ll`` per
    compiled module. That includes the runtime-bitcode side-module
    (``atomic_exchange_i32`` / ``atomic_add_i32`` and friends in
    ``runtime/llvm/runtime_module/atomic.h``), which is built from C++
    ``__atomic_*`` intrinsics under ``memory_order_seq_cst``. Those
    helpers are not user-kernel atomics and are not reachable from any
    of the kernels these tests compile, so a bare ``seq_cst atomicrmw``
    inside them must not be confused with a regression in the AMDGPU
    codegen contract. We identify the runtime-bitcode dump by its
    LLVM ``ModuleID`` header and skip it."""
    ll_files = sorted(dump_dir.glob("quadrants_kernel_amdgpu_llvm_ir_*.ll"))
    assert ll_files, f"no LLVM IR dumps produced in {dump_dir}"
    kernel_irs = []
    for p in ll_files:
        text = p.read_text()
        # The LLVM ``ModuleID`` header is always the first line of an
        # IR dump, e.g. ``; ModuleID = 'kernel'`` or
        # ``; ModuleID = 'runtime_bitcode'``.
        first_line = text.split("\n", 1)[0]
        if _RUNTIME_BITCODE_MODULE_ID in first_line:
            continue
        kernel_irs.append(text)
    assert kernel_irs, f"no user-kernel LLVM IR dumps in {dump_dir} " f"(only runtime-bitcode dumps were produced)"
    return "\n".join(kernel_irs)


def _run_kernel_dump(tmp_path: pathlib.Path, child_name: str) -> str:
    """Run the named child entry point in a subprocess with IR dumping
    enabled and return the concatenated IR text."""
    cmd = [sys.executable, __file__, child_name, str(tmp_path)]
    env = dict(os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + "."
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != RET_SUCCESS:
        print(proc.stdout)
        print("-" * 80)
        print(proc.stderr)
    assert proc.returncode == RET_SUCCESS, f"child '{child_name}' failed with exit code {proc.returncode}"
    return _read_all_ll(tmp_path)


def _init_amdgpu_with_dump(dump_dir: str) -> None:
    os.chdir(dump_dir)  # print_kernel_llvm_ir writes to CWD
    qd.init(arch=qd.amdgpu, print_kernel_llvm_ir=True, offline_cache=False)


# ---------------------------------------------------------------------------
# Per-op child entry points. Kept tiny so the IR dump is small and the
# atomicrmw / cmpxchg of interest is easy to grep.
# ---------------------------------------------------------------------------


def _child_atomic_add_i32(args):
    _init_amdgpu_with_dump(args[0])
    c = qd.field(qd.i32, shape=())

    @qd.kernel
    def k():
        for _ in range(16):
            qd.atomic_add(c[None], 1)

    k()
    sys.exit(RET_SUCCESS)


def _child_atomic_add_f32(args):
    _init_amdgpu_with_dump(args[0])
    c = qd.field(qd.f32, shape=())

    @qd.kernel
    def k():
        for _ in range(16):
            qd.atomic_add(c[None], 1.0)

    k()
    sys.exit(RET_SUCCESS)


def _child_atomic_or_i32(args):
    _init_amdgpu_with_dump(args[0])
    flags = qd.field(qd.i32, shape=())

    @qd.kernel
    def k():
        for i in range(16):
            qd.atomic_or(flags[None], 1 << (i % 8))

    k()
    sys.exit(RET_SUCCESS)


def _child_atomic_mul_i32(args):
    _init_amdgpu_with_dump(args[0])
    v = qd.field(qd.i32, shape=())
    v[None] = 1

    @qd.kernel
    def k():
        for i in range(1, 4):
            qd.atomic_mul(v[None], i)

    k()
    sys.exit(RET_SUCCESS)


def _child_atomic_mul_f32(args):
    _init_amdgpu_with_dump(args[0])
    v = qd.field(qd.f32, shape=())
    v[None] = 1.0

    @qd.kernel
    def k():
        for i in range(1, 4):
            qd.atomic_mul(v[None], qd.cast(i, qd.f32))

    k()
    sys.exit(RET_SUCCESS)


def _child_atomic_min_f32(args):
    _init_amdgpu_with_dump(args[0])
    v = qd.field(qd.f32, shape=())
    v[None] = 1e9

    @qd.kernel
    def k():
        for i in range(1, 16):
            qd.atomic_min(v[None], qd.cast(i, qd.f32))

    k()
    sys.exit(RET_SUCCESS)


def _child_atomic_max_f64(args):
    _init_amdgpu_with_dump(args[0])
    v = qd.field(qd.f64, shape=())
    v[None] = -1e18

    @qd.kernel
    def k():
        for i in range(1, 16):
            qd.atomic_max(v[None], qd.cast(i, qd.f64))

    k()
    sys.exit(RET_SUCCESS)


_CHILDREN = {
    fn.__name__: fn
    for fn in [
        _child_atomic_add_i32,
        _child_atomic_add_f32,
        _child_atomic_or_i32,
        _child_atomic_mul_i32,
        _child_atomic_mul_f32,
        _child_atomic_min_f32,
        _child_atomic_max_f64,
    ]
}


# ---------------------------------------------------------------------------
# Assertion helpers.
# ---------------------------------------------------------------------------


def _assert_no_system_seqcst_atomic(ir: str) -> None:
    """All atomic instructions on AMDGPU should be agent+monotonic. Any
    occurrence of bare ``seq_cst`` on an ``atomicrmw`` / ``cmpxchg`` is a
    regression."""
    bad = _SYSTEM_SEQCST_RMW_RE.findall(ir)
    assert not bad, f"unexpected seq_cst atomicrmw in AMDGPU IR: {bad[:3]}"
    bad = _SYSTEM_SEQCST_CMPXCHG_RE.findall(ir)
    assert not bad, f"unexpected seq_cst cmpxchg in AMDGPU IR: {bad[:3]}"


def _assert_has_agent_monotonic_rmw(ir: str) -> None:
    assert _ANY_RMW_RE.search(ir), "expected at least one atomicrmw in IR"
    assert _AGENT_MONO_RMW_RE.search(ir), (
        'no atomicrmw carries syncscope("agent") monotonic; '
        "atomic was emitted with stronger ordering or different scope"
    )


def _assert_has_agent_monotonic_cmpxchg(ir: str) -> None:
    assert _ANY_CMPXCHG_RE.search(ir), "expected at least one cmpxchg in IR"
    assert _AGENT_MONO_CMPXCHG_RE.search(ir), (
        'no cmpxchg carries syncscope("agent") monotonic; '
        "CAS-loop was emitted with stronger ordering or different scope"
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.amdgpu])
def test_ir_atomic_add_i32_is_agent_monotonic(tmp_path: pathlib.Path):
    ir = _run_kernel_dump(tmp_path, _child_atomic_add_i32.__name__)
    _assert_has_agent_monotonic_rmw(ir)
    _assert_no_system_seqcst_atomic(ir)


@test_utils.test(arch=[qd.amdgpu])
def test_ir_atomic_add_f32_is_agent_monotonic(tmp_path: pathlib.Path):
    ir = _run_kernel_dump(tmp_path, _child_atomic_add_f32.__name__)
    _assert_has_agent_monotonic_rmw(ir)
    _assert_no_system_seqcst_atomic(ir)


@test_utils.test(arch=[qd.amdgpu])
def test_ir_atomic_or_i32_is_agent_monotonic(tmp_path: pathlib.Path):
    ir = _run_kernel_dump(tmp_path, _child_atomic_or_i32.__name__)
    _assert_has_agent_monotonic_rmw(ir)
    _assert_no_system_seqcst_atomic(ir)


@test_utils.test(arch=[qd.amdgpu])
def test_ir_atomic_mul_i32_is_agent_monotonic_cmpxchg(tmp_path: pathlib.Path):
    # atomic_mul on int has no native AtomicRMW op -> CAS loop.
    ir = _run_kernel_dump(tmp_path, _child_atomic_mul_i32.__name__)
    _assert_has_agent_monotonic_cmpxchg(ir)
    _assert_no_system_seqcst_atomic(ir)


@test_utils.test(arch=[qd.amdgpu])
def test_ir_atomic_mul_f32_is_agent_monotonic_cmpxchg(tmp_path: pathlib.Path):
    ir = _run_kernel_dump(tmp_path, _child_atomic_mul_f32.__name__)
    _assert_has_agent_monotonic_cmpxchg(ir)
    _assert_no_system_seqcst_atomic(ir)


# ---------------------------------------------------------------------------
# f32/f64 min/max — Phase 3.1 routes these through the overridden
# atomic_op_using_cas path so they pick up agent+monotonic from the same
# default_atomic_ordering() / default_atomic_scope() virtuals as the
# integer paths. Asserted strictly.
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.amdgpu])
def test_ir_atomic_min_f32_is_agent_monotonic(tmp_path: pathlib.Path):
    ir = _run_kernel_dump(tmp_path, _child_atomic_min_f32.__name__)
    _assert_no_system_seqcst_atomic(ir)
    _assert_has_agent_monotonic_cmpxchg(ir)


@test_utils.test(arch=[qd.amdgpu])
def test_ir_atomic_max_f64_is_agent_monotonic(tmp_path: pathlib.Path):
    ir = _run_kernel_dump(tmp_path, _child_atomic_max_f64.__name__)
    _assert_no_system_seqcst_atomic(ir)
    _assert_has_agent_monotonic_cmpxchg(ir)


# ---------------------------------------------------------------------------
# Cross-cutting: a kernel that exercises both atomic_add and atomic_min
# on the same f32 field. Pre-Phase-3.1 this produced mixed-scope IR
# (agent+monotonic atomicrmw plus SeqCst+system runtime helper call).
# Post-Phase-3.1 both routes through default_atomic_*() and the IR
# contains only agent+monotonic atomics.
# ---------------------------------------------------------------------------


def _child_atomic_add_then_min_f32(args):
    _init_amdgpu_with_dump(args[0])
    v = qd.field(qd.f32, shape=())
    v[None] = 0.0

    @qd.kernel
    def k_add():
        for _ in range(16):
            qd.atomic_add(v[None], 1.0)

    @qd.kernel
    def k_min():
        for i in range(1, 16):
            qd.atomic_min(v[None], qd.cast(i, qd.f32))

    k_add()
    k_min()
    sys.exit(RET_SUCCESS)


_CHILDREN[_child_atomic_add_then_min_f32.__name__] = _child_atomic_add_then_min_f32


@test_utils.test(arch=[qd.amdgpu])
def test_ir_no_mixed_scope_on_same_field(tmp_path: pathlib.Path):
    ir = _run_kernel_dump(tmp_path, _child_atomic_add_then_min_f32.__name__)
    _assert_no_system_seqcst_atomic(ir)
    # Both atomic shapes should appear (atomicrmw for the add, cmpxchg
    # for the min CAS loop) and both should carry agent+monotonic.
    _assert_has_agent_monotonic_rmw(ir)
    _assert_has_agent_monotonic_cmpxchg(ir)


# ---------------------------------------------------------------------------
# Subprocess dispatch (mirrors test_fn_attrs.py).
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    name, *rest = sys.argv[1:]
    _CHILDREN[name](rest)
