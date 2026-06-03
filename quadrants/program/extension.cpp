#include "extension.h"

#include <unordered_map>
#include <unordered_set>

namespace quadrants::lang {

bool is_extension_supported(Arch arch, Extension ext) {
  static std::unordered_map<Arch, std::unordered_set<Extension>> arch2ext = {
      {Arch::x64,
       {Extension::sparse, Extension::quant, Extension::quant_basic, Extension::data64, Extension::adstack,
        Extension::assertion, Extension::extfunc, Extension::mesh}},
      {Arch::arm64,
       {Extension::sparse, Extension::quant, Extension::quant_basic, Extension::data64, Extension::adstack,
        Extension::assertion, Extension::mesh}},
      {Arch::cuda,
       {Extension::sparse, Extension::quant, Extension::quant_basic, Extension::data64, Extension::adstack,
        Extension::bls, Extension::assertion, Extension::mesh}},
      // NOTE(amdgpu): keep Extension::assertion only. The BLS test suite (test_bls.py,
      // test_bls_assume_in_range.py) uses `qd.root.pointer(...).dense(...)` which requires
      // Extension::sparse — a feature that has NOT been ported to AMDGPU. Declaring `bls`
      // (or quant/quant_basic/data64/adstack) here would make those tests RUN and fail with
      // "Pointer SNode is not supported on this backend" at field-builder time, instead of
      // cleanly skipping at the `@test_utils.test(require=qd.extension.bls)` decorator.
      // Re-add the relevant Extensions once their AMDGPU codegen lands.
      {Arch::amdgpu, {Extension::assertion}},
      {Arch::metal, {Extension::adstack}},
      {Arch::vulkan, {Extension::adstack}},
  };
  const auto &exts = arch2ext[arch];
  return exts.find(ext) != exts.end();
}

}  // namespace quadrants::lang
