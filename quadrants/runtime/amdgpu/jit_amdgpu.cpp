#include "quadrants/runtime/amdgpu/jit_amdgpu.h"
#include "quadrants/runtime/llvm/llvm_context.h"
#include "quadrants/runtime/llvm/llvm_context_pass.h"

#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <fstream>
#include <cstdlib>
#include <set>
#include <sstream>
#include <vector>

namespace quadrants {
namespace lang {
#if defined(QD_WITH_AMDGPU)
JITModule *JITSessionAMDGPU ::add_module(std::unique_ptr<llvm::Module> M,
                                         int max_reg) {
  auto hsaco = compile_module_to_hsaco(M);
  QD_TRACE("hsaco size: {:.2f}KB", hsaco.size() / 1024.0);

  void *amdgpu_module;
  auto t = Time::get_time();
  AMDGPUDriver::get_instance().module_load_data(&amdgpu_module, hsaco.c_str());
  QD_TRACE("AMDGPU load data from module time : {}ms",
           (Time::get_time() - t) * 1000);
  modules.push_back(std::make_unique<JITModuleAMDGPU>(amdgpu_module));
  return modules.back().get();
}

std::string JITSessionAMDGPU::compile_module_to_hsaco(
    std::unique_ptr<llvm::Module> &llvm_module) {
  // Phase 1: Convert allocas to addrspace(5) with addrspacecast to flat
  {
    llvm::legacy::FunctionPassManager fpm(llvm_module.get());
    fpm.add(new AMDGPUConvertAllocaInstAddressSpacePass());
    fpm.doInitialization();
    for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func)
      fpm.run(*func);
    fpm.doFinalization();
  }

  if (llvm::verifyModule(*llvm_module, &llvm::errs())) {
    llvm_module->print(llvm::errs(), nullptr);
    QD_WARN("Module broken");
  }
  using namespace llvm;

  if (this->config_.print_kernel_llvm_ir) {
    static FileSequenceWriter writer(
        "quadrants_kernel_amdgpu_llvm_ir_{:04d}.ll",
        "unoptimized LLVM IR (AMDGPU)");
    writer.write(llvm_module.get());
  }
  auto triple_str = llvm_module->getTargetTriple();
  std::string error_str;
  auto target = llvm::TargetRegistry::lookupTarget(triple_str, error_str);

  llvm::TargetOptions options;
  options.MCOptions.AsmVerbose = false;
  if (this->config_.fast_math) {
    options.AllowFPOpFusion = FPOpFusion::Fast;
  } else {
    options.AllowFPOpFusion = FPOpFusion::Strict;
  }
  options.NoZerosInBSS = 0;
  options.GuaranteedTailCallOpt = 0;

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      triple_str, AMDGPUContext::get_instance().get_mcpu(), "", options,
      llvm::Reloc::PIC_, llvm::CodeModel::Small,
      llvm::CodeGenOptLevel::Aggressive));

  llvm_module->setDataLayout(machine->createDataLayout());

  if (this->config_.print_kernel_amdgcn) {
    // Amdgcn will not generated during generating hsaco file
    // It's an interim impl
    // while add machine info to pass_manager, the module(LLVM-IR) will add more
    // target-specific info e.g.
    //   call { i1, i32 } @llvm.amdgcn.if.i32(i1 %15)
    // then then `addPassesToEmitFile` will occur an error
    //   LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.if
    // related https://github.com/llvm/llvm-project/issues/60727
    //    we can't though the `addPassesToEmitFile` to generate GCN file
    //    directly
    // another way
    //    llvm-objdump -d xxxx.hsaco(can ensure that hsaco and gcn correspond to
    //    each other)

    auto module_clone = llvm::CloneModule(*llvm_module);
    llvm::legacy::PassManager module_gen_gcn_pass_manager;
    llvm::SmallString<0> gcnstr;
    llvm::raw_svector_ostream llvm_stream_gcn(gcnstr);
    std::unique_ptr<llvm::TargetMachine> machine_gen_gcn(
        target->createTargetMachine(
            triple_str, AMDGPUContext::get_instance().get_mcpu(), "", options,
            llvm::Reloc::PIC_, llvm::CodeModel::Small,
            llvm::CodeGenOptLevel::Aggressive));

    // Replace PassManagerBuilder with PassBuilder API
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::PassBuilder pb(machine_gen_gcn.get());
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm =
        pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    mpm.run(*module_clone, mam);

    module_gen_gcn_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(
        machine_gen_gcn->getTargetIRAnalysis()));
    machine_gen_gcn->addPassesToEmitFile(
        module_gen_gcn_pass_manager, llvm_stream_gcn, nullptr,
        llvm::CodeGenFileType::AssemblyFile, true);
    module_gen_gcn_pass_manager.run(*module_clone);
    std::string gcn(gcnstr.begin(), gcnstr.end());
    static FileSequenceWriter writer("quadrants_kernel_amdgcn_{:04d}.gcn",
                                     "module AMDGCN");
    writer.write(gcn);
  }

  auto tmp_dir = get_tmp_dir();
  uint64 random_num = get_random_num();

  auto ll_filename = "quadrants_amdgcn_" + std::to_string(random_num) + ".ll";
  auto hsaco_filename =
      "quadrants_amdgcn_" + std::to_string(random_num) + ".hsaco";
  auto ll_path = tmp_dir + ll_filename;
  auto hsaco_path = tmp_dir + hsaco_filename;

  // Write unoptimized LLVM IR to disk
  {
    std::error_code ec;
    llvm::raw_fd_ostream ll_stream(ll_path, ec);
    if (ec)
      QD_ERROR("Failed to open {} for writing: {}", ll_path, ec.message());
    llvm_module->print(ll_stream, nullptr);
  }

  // Patch kernel attributes: remove amdgpu-no-agpr to allow AGPR spilling
  // waves-per-eu and flat-work-group-size are set per-kernel in C++
  {
    std::string sed_cmd;
    sed_cmd = "sed -i 's/\"amdgpu-no-agpr\" //g' " + ll_path;
    std::system(sed_cmd.c_str());
  }

  QD_TRACE("Compiling module via external clang...");
  [[maybe_unused]] auto _ = AMDGPUContext::get_instance().get_lock_guard();

  // Use QD_CLANG env var, fall back to ROCM_PATH/llvm/bin/clang, then amdclang++
  std::string clang_executable;
  const char *qd_clang = std::getenv("QD_CLANG");
  const char *rocm_path = std::getenv("ROCM_PATH");
  if (qd_clang) {
    clang_executable = qd_clang;
  } else if (rocm_path) {
    clang_executable = std::string(rocm_path) + "/llvm/bin/clang";
  } else {
    clang_executable = "clang";
  }

  auto mcpu = AMDGPUContext::get_instance().get_mcpu();
  std::string fast_math_flags = this->config_.fast_math
      ? "-ffast-math -ffp-contract=fast"
      : "-ffp-contract=off";

  auto asm_filename = "quadrants_amdgcn_" + std::to_string(random_num) + ".s";
  auto asm_path = tmp_dir + asm_filename;

  // Step 1: Compile LLVM IR to assembly via clang -O3
  std::string compile_cmd = clang_executable +
      " -x ir"
      " -target amdgcn-amd-amdhsa"
      " -mcpu=" + mcpu +
      " -O3"
      " " + fast_math_flags +
      " -nogpulib"
      " -mllvm -amdgpu-spill-vgpr-to-agpr=1"
      " -mllvm -unroll-threshold=100"
      " -S"
      " -o " + asm_path +
      " " + ll_path;

  QD_TRACE("Compiling to assembly: {}", compile_cmd);
  if (std::system(compile_cmd.c_str()))
    QD_ERROR(
        fmt::format("Clang compilation to assembly failed for {}. Command: {}",
                    asm_filename, compile_cmd));

  // Step 2: Per-function flat_load/flat_store -> global_load/global_store.
  // A function is skipped if it:
  //   - Uses LDS (ds_read/ds_write): flat accesses may target shared memory
  //   - Has uses_flat_scratch=1: a private pointer escapes to generic address
  //     space, so flat accesses may dereference private memory
  {
    std::ifstream asm_file(asm_path);
    std::string asm_content((std::istreambuf_iterator<char>(asm_file)),
                             std::istreambuf_iterator<char>());
    asm_file.close();

    std::vector<std::string> lines;
    {
      std::istringstream stream(asm_content);
      std::string line;
      while (std::getline(stream, line))
        lines.push_back(line);
    }

    // Pass 1: Collect unsafe function names.
    std::set<std::string> unsafe_funcs;

    // Functions with uses_flat_scratch=1 (private pointer escape)
    for (const auto& l : lines) {
      if (l.find(".set") != std::string::npos &&
          l.find(".uses_flat_scratch, 1") != std::string::npos) {
        auto dot_l = l.find(".L");
        if (dot_l != std::string::npos) {
          auto ufs = l.find(".uses_flat_scratch", dot_l);
          if (ufs != std::string::npos)
            unsafe_funcs.insert(l.substr(dot_l + 2, ufs - dot_l - 2));
        }
      }
    }

    // Functions with LDS usage (ds_read/ds_write in their body)
    {
      std::string cur_func;
      for (const auto& l : lines) {
        if (!l.empty() && l[0] != '\t' && l[0] != ' ' && l[0] != '.' &&
            l[0] != ';' && l[0] != '#') {
          auto colon = l.find(':');
          if (colon != std::string::npos)
            cur_func = l.substr(0, colon);
        }
        if (!cur_func.empty() &&
            (l.find("ds_read") != std::string::npos ||
             l.find("ds_write") != std::string::npos)) {
          unsafe_funcs.insert(cur_func);
        }
      }
    }

    for (const auto& name : unsafe_funcs)
      QD_TRACE("  skip flat-to-global for: {} (LDS or flat scratch)", name);

    // Pass 2: Convert flat_load/flat_store only within safe functions.
    bool modified = false;
    int converted_count = 0;
    int skipped_count = 0;
    std::string cur_func;
    bool cur_safe = false;

    for (auto& l : lines) {
      // Detect function start: label at column 0
      if (!l.empty() && l[0] != '\t' && l[0] != ' ' && l[0] != '.' &&
          l[0] != ';' && l[0] != '#') {
        auto colon = l.find(':');
        if (colon != std::string::npos) {
          cur_func = l.substr(0, colon);
          cur_safe = unsafe_funcs.find(cur_func) == unsafe_funcs.end();
        }
      }

      // Find flat_load_ or flat_store_ instruction
      auto flat_pos = l.find("flat_load_");
      if (flat_pos == std::string::npos)
        flat_pos = l.find("flat_store_");

      if (flat_pos == std::string::npos || flat_pos == 0 ||
          (l[flat_pos - 1] != '\t' && l[flat_pos - 1] != ' '))
        continue;

      // Skip if inside a comment
      auto comment_pos = l.find(';');
      if (comment_pos != std::string::npos && comment_pos < flat_pos)
        continue;

      if (!cur_safe) {
        skipped_count++;
        continue;
      }

      // Replace "flat_" with "global_" (5 chars -> 7 chars)
      l.replace(flat_pos, 5, "global_");

      // Append ", off" before " offset:" or at end of line
      auto offset_pos = l.find(" offset:");
      if (offset_pos != std::string::npos) {
        l.insert(offset_pos, ", off");
      } else {
        auto last = l.find_last_not_of(" \t\r\n");
        if (last != std::string::npos) {
          l.erase(last + 1);
          l += ", off";
        }
      }
      modified = true;
      converted_count++;
    }

    QD_TRACE("flat-to-global: {} instructions converted, {} skipped (unsafe)",
             converted_count, skipped_count);

    if (modified) {
      std::ofstream asm_out(asm_path);
      for (const auto& l : lines)
        asm_out << l << '\n';
    }
  }

  // Step 3: Assemble patched assembly + link to HSACO
  std::string assemble_cmd = clang_executable +
      " -x assembler"
      " -target amdgcn-amd-amdhsa"
      " -mcpu=" + mcpu +
      " -Xlinker --no-undefined"
      " -o " + hsaco_path +
      " " + asm_path;

  QD_TRACE("Assembling to HSACO: {}", assemble_cmd);
  if (std::system(assemble_cmd.c_str()))
    QD_ERROR(
        fmt::format("Assembly to HSACO failed for {}. Command: {}",
                    hsaco_filename, assemble_cmd));

  std::string hsaco_str = load_hsaco(hsaco_path);

  if (this->config_.print_kernel_llvm_ir_optimized) {
    // With external clang, dump the optimized IR via a separate opt call
    auto opt_ll_path = tmp_dir + "quadrants_amdgcn_" +
        std::to_string(random_num) + "_optimized.ll";
    std::string opt_executable;
    auto slash_pos = clang_executable.rfind('/');
    if (slash_pos != std::string::npos) {
      opt_executable = clang_executable.substr(0, slash_pos) + "/opt";
    } else {
      opt_executable = "opt";
    }
    std::string opt_cmd = opt_executable +
        " -O3 -S -o " + opt_ll_path + " " + ll_path;
    if (std::system(opt_cmd.c_str()) == 0) {
      std::ifstream opt_ll_file(opt_ll_path);
      std::string opt_ll_content(
          (std::istreambuf_iterator<char>(opt_ll_file)),
          std::istreambuf_iterator<char>());
      static FileSequenceWriter writer(
          "quadrants_kernel_amdgpu_llvm_ir_optimized_{:04d}.ll",
          "optimized LLVM IR (AMDGPU)");
      writer.write(opt_ll_content);
    }
  }

  return hsaco_str;
}

std::unique_ptr<JITSession> create_llvm_jit_session_amdgpu(
    QuadrantsLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  QD_ASSERT(arch == Arch::amdgpu);
  auto data_layout = QuadrantsLLVMContext::get_data_layout(arch);
  return std::make_unique<JITSessionAMDGPU>(tlctx, config, data_layout);
}
#else
std::unique_ptr<JITSession> create_llvm_jit_session_amdgpu(
    QuadrantsLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  QD_NOT_IMPLEMENTED
}
#endif

}  // namespace lang
}  // namespace quadrants
