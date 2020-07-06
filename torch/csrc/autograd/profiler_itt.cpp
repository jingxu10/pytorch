#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/itt_wrapper.h>

namespace torch { namespace autograd { namespace profiler {

namespace {
struct ITTMethods : public ITTStubs {
  void ittMark(const char* name) override {
	torch::itt_mark(name);
  }
  void ittRangePush(const char* name) override {
    torch::itt_range_push(name);
  }
  void ittRangePop() override {
    torch::itt_range_pop();
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
