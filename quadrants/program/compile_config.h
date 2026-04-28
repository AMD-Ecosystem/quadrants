#pragma once

#include "quadrants/rhi/arch.h"
#include "quadrants/util/lang_util.h"

namespace quadrants::lang {

struct CompileConfig {
  Arch arch;
  bool debug;
  bool cfg_optimization;
  bool check_out_of_bound;
  bool validate_autodiff;
  int simd_width;
  int opt_level;
  int external_optimization_level;
  int max_vector_width;
  bool raise_on_templated_floats{false};
  bool print_preprocessed_ir;
  bool print_ir;
  bool print_accessor_ir;
  bool print_ir_dbg_info;
  bool serial_schedule;
  bool simplify_before_lower_access;
  bool lower_access;
  bool simplify_after_lower_access;
  bool move_loop_invariant_outside_if;
  bool cache_loop_invariant_global_vars{true};
  bool demote_dense_struct_fors;
  bool advanced_optimization;
  bool constant_folding;
  bool use_llvm;
  bool verbose_kernel_launches;
  bool kernel_profiler;
  bool timeline{false};
  bool verbose;
  bool fast_math;
  bool flatten_if;
  bool make_thread_local;
  bool make_block_local;
  bool detect_read_only;
  bool real_matrix_scalarize;
  bool force_scalarize_matrix;
  bool half2_vectorization;
  bool make_cpu_multithreading_loop;
  DataType default_fp;
  DataType default_ip;
  DataType default_up;
  std::string extra_flags;
  int default_cpu_block_dim;
  bool cpu_block_dim_adaptive;
  int default_gpu_block_dim;
  int gpu_max_reg;
  bool ad_stack_experimental_enabled{false};
  int ad_stack_size{0};  // 0 = adaptive
  // The default size when the Quadrants compiler is unable to automatically
  // determine the autodiff stack size.
  int default_ad_stack_size{32};

  int saturating_grid_dim;
  int max_block_dim;
  int cpu_max_num_threads;
  int random_seed;

  // Debugging options:
  bool print_struct_llvm_ir;
  bool print_kernel_llvm_ir;
  bool print_kernel_llvm_ir_optimized;
  bool print_kernel_asm;
  bool print_kernel_amdgcn;
  std::string debug_dump_path{"/tmp/ir/"};

  // CUDA/AMDGPU backend options:
  float64 device_memory_GB;
  float64 device_memory_fraction;

  bool quant_opt_store_fusion{true};
  bool quant_opt_atomic_demotion{true};

  // Mesh related.
  // MeshQuadrants options
  bool make_mesh_block_local{true};
  bool optimize_mesh_reordered_mapping{true};
  bool mesh_localize_to_end_mapping{true};
  bool mesh_localize_from_end_mapping{false};
  bool mesh_localize_all_attr_mappings{false};
  bool demote_no_access_mesh_fors{true};
  bool experimental_auto_mesh_local{false};
  int auto_mesh_local_default_occupacy{4};

  // Offline cache options
  bool offline_cache{false};
  std::string offline_cache_file_path{get_repo_dir() + "qdcache"};
  std::string offline_cache_cleaning_policy{
      "lru"};  // "never"|"version"|"lru"|"fifo"
  int offline_cache_max_size_of_files{100 * 1024 *
                                      1024};   // bytes, default: 100MB
  double offline_cache_cleaning_factor{0.25};  // [0.f, 1.f]

  int num_compile_threads{4};
  std::string vk_api_version;

  size_t cuda_stack_limit{0};

  // Selects the AMDGPU "amdgpu-waves-per-eu" attribute applied to heavy
  // kernels in jit_amdgpu.cpp:
  //   false (default): legacy "1,2" budget — permits VGPR-heavy,
  //                    low-occupancy codegen. Wins for kernels whose
  //                    inlined call chain genuinely needs >256 archived
  //                    registers/wave (e.g. constraint solver monolith).
  //   true:            "4,8" budget — forces fewer VGPRs/wave so more
  //                    waves co-reside per SIMD. Helpful for memory-
  //                    latency-bound kernels with small live-range graphs,
  //                    but pays heavy scratch-spill traffic on register-
  //                    heavy kernels and can regress overall perf.
  // Override from Python (qd.init(amdgpu_auto_waves_per_eu=True)) or env
  // var (QD_AMDGPU_AUTO_WAVES_PER_EU=1).
  bool amdgpu_auto_waves_per_eu{false};

  // When true, AMDGPU global-memory loads (those that the
  // AMDGPUFlatToGlobalLoadStorePass converts from addrspace(0) flat to
  // addrspace(1) global) are tagged with LLVM !nontemporal metadata.
  // The AMDGPU backend lowers this to a cache hint that biases the load
  // away from the L1 vector cache (16 KB / CU on gfx942), which is
  // useful when the kernel's working set is many times the L1 size and
  // L1 mostly serves as a pollution channel rather than a true cache.
  // Helpful for memory-latency-bound kernels reading large data
  // structures (e.g., the constraint solver's 181 MB Jacobian on
  // 8192-batch G1 sims). May regress kernels whose hot-set genuinely
  // fits in L1.
  // Toggle from Python (qd.init(amdgpu_nontemporal_global_loads=True))
  // or env var (QD_AMDGPU_NONTEMPORAL_GLOBAL_LOADS=1).
  bool amdgpu_nontemporal_global_loads{false};

  // Same idea as amdgpu_nontemporal_global_loads, but for global
  // stores. The hint asks the cache hierarchy to write through rather
  // than allocating an L1 line, freeing L1 for read traffic. Useful
  // when stored values won't be re-read soon by the same wave (e.g.,
  // per-iter result writes that are read again only in the next solver
  // iteration after the L1 line would have been evicted anyway).
  // Toggle from Python (qd.init(amdgpu_nontemporal_global_stores=True))
  // or env var (QD_AMDGPU_NONTEMPORAL_GLOBAL_STORES=1).
  bool amdgpu_nontemporal_global_stores{false};

  CompileConfig();

  void fit();
};

extern QD_DLL_EXPORT CompileConfig default_compile_config;

}  // namespace quadrants::lang
