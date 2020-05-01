#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/intel/itt_wrapper.h>

namespace torch { namespace autograd { namespace profiler {

namespace {
struct ITTMethods : public ITTStubs {
  void ittMark(const char* name) override {
	torch::intel::itt_mark(name);
  }
  void ittRangePush(const char* name) override {
    torch::intel::itt_range_push(name);
  }
  void ittRangePop() override {
    torch::intel::itt_range_pop();
  }
  bool enabled() override {
    return true;
  }
};

struct RegisterITTMethods {
  RegisterITTMethods() {
    static ITTMethods methods;
    registerITTMethods(&methods);
  }
};
RegisterITTMethods reg;

} // namespaces
} // namespace profiler
} // namespace autograd
} // namespace torch
