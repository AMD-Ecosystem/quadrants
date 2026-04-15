#include "quadrants/runtime/amdgpu/jit_amdgpu.h"
#include "quadrants/runtime/llvm/llvm_context.h"
#include "quadrants/runtime/llvm/llvm_context_pass.h"

#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <fstream>
#include <cstdlib>

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
  llvm::legacy::FunctionPassManager function_pass_manager_addrcast(
      llvm_module.get());
  function_pass_manager_addrcast.add(
      new AMDGPUConvertAllocaInstAddressSpacePass());
  function_pass_manager_addrcast.doInitialization();
  for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func)
    function_pass_manager_addrcast.run(*func);
  function_pass_manager_addrcast.doFinalization();

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

  std::string clang_cmd = clang_executable +
      " -x ir"
      " -target amdgcn-amd-amdhsa"
      " -mcpu=" + mcpu +
      " -O3"
      " " + fast_math_flags +
      " -nogpulib"
      " -mllvm -amdgpu-spill-vgpr-to-agpr=1"
      " -mllvm -unroll-threshold=100"
      " -Xlinker --no-undefined"
      " -o " + hsaco_path +
      " " + ll_path;

  QD_TRACE("Compiling with command: {}", clang_cmd);
  if (std::system(clang_cmd.c_str()))
    QD_ERROR(
        fmt::format("Clang compilation failed for {}. Command: {}",
                    hsaco_filename, clang_cmd));

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
