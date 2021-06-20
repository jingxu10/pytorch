#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>


#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_prelu(const Tensor& input, const Tensor& weight) {
  TORCH_CHECK(false, "mkldnn_prelu: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_prelu_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight) {
  TORCH_CHECK(false, "mkldnn_prelu_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at { namespace native {

Tensor mkldnn_prelu(const Tensor& input, const Tensor& weight) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_prelu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  const ideep::tensor x = itensor_from_mkldnn(input);
  const ideep::tensor w = itensor_from_tensor(weight);
  ideep::tensor y;
  ideep::prelu_forward::compute(x, w, y);
  return new_with_itensor_mkldnn(std::move(y),
		  optTypeMetaToScalarType(input.options().dtype_opt()),
		  input.options().device_opt());
}

std::tuple<Tensor, Tensor> mkldnn_prelu_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight) {
  const ideep::tensor x = itensor_from_mkldnn(input);
  const ideep::tensor grady = itensor_from_mkldnn(grad_output);
  const ideep::tensor w = itensor_from_tensor(weight);
  ideep::tensor gradx;
  ideep::tensor gradw;
  ideep::prelu_backward::compute(x, w, grady, gradx, gradw);
  return std::tuple<Tensor, Tensor>{
      new_with_itensor_mkldnn(std::move(gradx),
             optTypeMetaToScalarType(input.options().dtype_opt()),
             input.options().device_opt()),
      new_with_itensor_mkldnn(std::move(gradw),
             optTypeMetaToScalarType(weight.options().dtype_opt()),
             weight.options().device_opt())
  };
}

}}

#endif // AT_MKLDNN_EBABLED
