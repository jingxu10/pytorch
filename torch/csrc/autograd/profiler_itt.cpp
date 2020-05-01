// #include <torch/csrc/utils/pybind.h>

#include "ittnotify.h"

namespace torch { namespace intel {
__itt_domain* _itt_domain = __itt_domain_create("PyTorch");

void itt_range_push(const char* msg) {
	__itt_string_handle* hsMsg = __itt_string_handle_create(msg);
	__itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
}

void itt_range_pop() {
	__itt_task_end(_itt_domain);
}

void itt_mark(const char* msg) {
	__itt_string_handle* hsMsg = __itt_string_handle_create(msg);
	__itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
	__itt_task_end(_itt_domain);
}

// void initIttBindings(PyObject* module) {
//   auto m = py::handle(module).cast<py::module>();
//
//   auto itt = m.def_submodule("_itt", "VTune ITT bindings");
//   itt.def("rangePush", itt_range_push);
//   itt.def("rangePop", itt_range_pop);
//   itt.def("mark", itt_mark);
// }
}} // namespace torch::intel
